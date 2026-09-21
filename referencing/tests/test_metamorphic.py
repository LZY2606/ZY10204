"""
Deformation-focused tests for the reference graph.

These tests build small, custom `Specification` objects (rather than only
JSON Schema documents) so that the relationship between

    * `Specification.id_of` / `Specification.subresources_of` /
      `Specification.anchors_in`,
    * `Resource` / `Registry` / `Resolver`, and
    * `Anchor` / `DynamicAnchor`

can be exercised under controlled "deformations": eager vs. lazy crawling,
unrelated resources, base-URI normalization, registry combination order,
opaque subtrees, and monkeypatched mutations of core invariants.

Nothing here touches the network, the filesystem, or real time.
"""

from __future__ import annotations

import pytest

from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DynamicAnchor

# ---------------------------------------------------------------------------
# Two small custom specifications
# ---------------------------------------------------------------------------


def _mapping(contents: object) -> dict:
    return contents if isinstance(contents, dict) else {}


def _tree_id(contents: object):
    return _mapping(contents).get("ID")


def _tree_subresources(contents: object) -> list:
    return list(_mapping(contents).get("children", ()))


def _tree_anchors(specification, contents: object):
    document = _mapping(contents)
    for name, target in document.get("anchors", {}).items():
        yield Anchor(
            name=name,
            resource=specification.create_resource(target),
        )
    for name, target in document.get("dynamicAnchors", {}).items():
        yield DynamicAnchor(
            name=name,
            resource=specification.create_resource(target),
        )


def _in_tree(segments, resolver, subresource):
    """
    Enter a subresource while JSON-pointer-walking along ``children``.

    Each entry is the keyword ``children``, each odd entry an integer index.
    """
    steps = iter(segments)
    for segment in steps:
        if segment != "children" or next(steps, None) is None:
            return resolver
    return resolver.in_subresource(subresource)


#: A specification whose documents form a hierarchy with both plain and
#: dynamic anchors, mirroring draft 2020-12 semantics on a smaller vocabulary.
HIERARCHICAL = Specification(
    name="hierarchical",
    id_of=_tree_id,
    subresources_of=_tree_subresources,
    anchors_in=_tree_anchors,
    maybe_in_subresource=_in_tree,
)


def _is_opaque(contents: object) -> bool:
    return bool(_mapping(contents).get("opaque"))


def _partial_id(contents: object):
    if _is_opaque(contents):
        return None
    return _mapping(contents).get("ID")


def _partial_subresources(contents: object) -> list:
    if _is_opaque(contents):
        return []
    return list(_mapping(contents).get("children", ()))


def _partial_anchors(specification, contents: object):
    if _is_opaque(contents):
        return []
    return _tree_anchors(specification, contents)


#: A specification where a document marked ``"opaque": true`` is an opaque
#: boundary: it has no ID, yields no subresources and contributes no anchors,
#: exactly like `Specification.OPAQUE` for the subtree rooted at it.
PARTIALLY_OPAQUE = Specification(
    name="partially-opaque",
    id_of=_partial_id,
    subresources_of=_partial_subresources,
    anchors_in=_partial_anchors,
    maybe_in_subresource=_in_tree,
)


# ---------------------------------------------------------------------------
# Shared corpora
# ---------------------------------------------------------------------------

# A three-level hierarchy with relative IDs and dynamic anchors at every
# level.  Each ``meta`` anchor points to a distinct document (with an
# absolute ID) so that resolving it is observable via the resulting base URI.
DYNAMIC_TREE = {
    "ID": "http://example.com/tree/",
    "anchors": {
        "plainRoot": {"where": "plain-root"},
        "plain Root": {"where": "spaced-root"},
    },
    "dynamicAnchors": {
        "meta": {"ID": "http://example.com/targets/root", "where": "root"},
    },
    "children": [
        {
            "ID": "child/",
            "anchors": {"plainChild": {"where": "plain-child"}},
            "dynamicAnchors": {
                "meta": {
                    "ID": "http://example.com/targets/child",
                    "where": "child",
                },
            },
            "children": [
                {
                    "ID": "leaf",
                    "dynamicAnchors": {
                        "meta": {
                            "ID": "http://example.com/targets/leaf",
                            "where": "leaf",
                        },
                    },
                    "children": [],
                },
            ],
        },
    ],
}

# The same tree, but with a non-dynamic (plain) anchor partway up the
# dynamic scope -- the dynamic resolution must not bind to it and must fall
# back to the original dynamic anchor.
DYNAMIC_TREE_WITH_GAP = {
    # The outermost resource deliberately defines no ``solo`` anchor at
    # all; the middle scope defines only a *plain* one, so the dynamic
    # reference can bind neither out there and must fall back to its origin.
    "ID": "http://example.com/gap/",
    "children": [
        {
            "ID": "child/",
            "anchors": {"solo": {"where": "plain-child"}},
            "children": [
                {
                    "ID": "leaf",
                    "dynamicAnchors": {
                        "solo": {
                            "ID": "http://example.com/gap-targets/leaf",
                            "where": "leaf",
                        },
                    },
                    "children": [],
                },
            ],
        },
    ],
}


# A hierarchy where one subtree is declared opaque.  Nothing below the
# opaque node (including ``hidden`` and its anchor) may be discoverable.
OPAQUE_TREE = {
    "ID": "http://example.com/opaque/",
    "anchors": {"visible": {"where": "root"}},
    "children": [
        {
            "ID": "normal/",
            "children": [
                {"ID": "deep", "anchors": {"deepAnchor": {"where": "deep"}}},
            ],
        },
        {
            "ID": "blackbox/",
            "opaque": True,
            # Everything below is deliberately not part of the graph:
            "anchors": {"secret": {"where": "secret"}},
            "children": [{"ID": "hidden"}],
        },
    ],
}


def _self_cycle():
    node = {"ID": "http://example.com/cycle/a"}
    node["children"] = [node]
    # Re-wrap the contents in a new resource so the root frame starts with an
    # empty ancestry; its child then re-enters the same object cyclically.
    root = HIERARCHICAL.create_resource(
        {"ID": "http://example.com/cycle/", "children": [node]},
    )
    return root, "http://example.com/cycle/a"


def _mutual_cycle():
    first = {"ID": "http://example.com/cycle/a"}
    second = {
        "ID": "http://example.com/cycle/b",
        "children": [first],
    }
    first["children"] = [second]
    root = {
        "ID": "http://example.com/cycle/",
        "children": [first],
    }
    return (
        HIERARCHICAL.create_resource(root),
        "http://example.com/cycle/a",
    )


# ---------------------------------------------------------------------------
# Dynamic anchor resolution: contents, resolved URI, scope order, fallback
# ---------------------------------------------------------------------------


def _descend_to_leaf(resolver):
    root = resolver.lookup("http://example.com/tree/")
    child = root.resolver.lookup("#/children/0")
    return child.resolver.lookup("leaf").resolver


class TestDynamicAnchors:
    def test_dynamic_resolution_contents_uri_and_scope(self):
        """
        A dynamic ref binds to the outermost dynamic anchor in scope, and
        along the way we observe the resolved base URI and full scope order.
        """
        resource = HIERARCHICAL.create_resource(DYNAMIC_TREE)
        registry = resource @ Registry()
        resolver = _descend_to_leaf(registry.resolver())

        # Sanity: the leaf itself is the current base.
        assert resolver._base_uri == "http://example.com/tree/child/leaf"

        # Dynamic scope is nearest-first ... outermost-last.
        scope = list(resolver.dynamic_scope())
        assert [uri for uri, _ in scope] == [
            "http://example.com/tree/child/",
            "http://example.com/tree/",
        ]
        # Each scope entry is paired with the same (crawled) registry.
        crawled = registry.crawl()
        assert [each_registry for _, each_registry in scope] == [
            crawled,
            crawled,
        ]

        resolved = resolver.lookup("#meta")
        assert resolved.contents == {
            "ID": "http://example.com/targets/root",
            "where": "root",
        }
        # The resolved URI is the bound anchor target's absolute ID, not the
        # leaf URI where the dynamic reference originated.
        assert resolved.resolver._base_uri == "http://example.com/targets/root"

    def test_dynamic_anchor_falls_back_when_scope_has_plain_anchor(self):
        """
        A plain (non-dynamic) anchor in an outer scope must not bind a
        dynamic reference; the original dynamic anchor is the fallback.
        """
        resource = HIERARCHICAL.create_resource(DYNAMIC_TREE_WITH_GAP)
        registry = resource @ Registry()

        root = registry.resolver().lookup("http://example.com/gap/")
        child = root.resolver.lookup("#/children/0")
        leaf = child.resolver.lookup("leaf").resolver

        assert [uri for uri, _ in leaf.dynamic_scope()] == [
            "http://example.com/gap/child/",
            "http://example.com/gap/",
        ]

        resolved = leaf.lookup("#solo")
        # The middle plain ``solo`` anchor is skipped (it is not dynamic)
        # and the outer scope has no ``solo`` anchor at all, so the
        # dynamic reference falls back to the leaf's own dynamic anchor.
        assert resolved.contents == {
            "ID": "http://example.com/gap-targets/leaf",
            "where": "leaf",
        }
        assert (
            resolved.resolver._base_uri
            == "http://example.com/gap-targets/leaf"
        )

    def test_same_named_plain_and_dynamic_anchor_prefers_dynamic(self):
        """
        When a single document defines both kinds of anchors with one name,
        crawling keeps the dynamic anchor, mirroring draft 2020-12.
        """
        document = {
            "ID": "http://example.com/both/",
            "anchors": {"both": {"where": "plain"}},
            "dynamicAnchors": {"both": {"where": "dynamic"}},
        }
        registry = HIERARCHICAL.create_resource(document) @ Registry()
        crawled = registry.crawl()
        anchor = crawled.anchor("http://example.com/both/", "both").value
        assert isinstance(anchor, DynamicAnchor)

        resolver = crawled.resolver("http://example.com/both/")
        assert resolver.lookup("#both").contents == {"where": "dynamic"}

    def test_trivial_dynamic_anchor_has_empty_scope_and_resolves_to_self(self):
        resource = HIERARCHICAL.create_resource(
            {"dynamicAnchors": {"meta": {"where": "only"}}},
        )
        resolver = (
            Registry().with_resource("http://example.com", resource).resolver()
        )
        resolved = resolver.lookup("http://example.com#meta")
        assert resolved.contents == {"where": "only"}
        assert list(resolved.resolver.dynamic_scope()) == []


# ---------------------------------------------------------------------------
# Metamorphic relations: deformations that must preserve lookup results
# ---------------------------------------------------------------------------


#: Equivalent reference formulations expressed as
#: ``(base_uri, reference, canonical_uri)``.  The base URIs are syntactically
#: different but URL-normalize to the same document, and the references are
#: resolved relative to them (or given absolutely with a relative ref).
EQUIVALENT_REFERENCES = [
    # ``(base_uri, reference, canonical_uri)``.  The reference variants all
    # normalize (via dot-segment resolution in the relative reference) to
    # the same canonical document.
    ("http://example.com/tree/", "", "http://example.com/tree/"),
    ("http://example.com/tree/", "./", "http://example.com/tree/"),
    ("http://example.com/tree/other/", "../", "http://example.com/tree/"),
    (
        "http://example.com/tree/other/",
        "../../tree/",
        "http://example.com/tree/",
    ),
    # Leaf-level equivalences.
    (
        "http://example.com/tree/",
        "child/leaf",
        "http://example.com/tree/child/leaf",
    ),
    (
        "http://example.com/tree/",
        "./child/leaf",
        "http://example.com/tree/child/leaf",
    ),
    (
        "http://example.com/tree/",
        "child/../child/leaf",
        "http://example.com/tree/child/leaf",
    ),
    (
        "http://example.com/tree/",
        "../tree/child/leaf",
        "http://example.com/tree/child/leaf",
    ),
    (
        "http://example.com/tree/child/",
        "leaf",
        "http://example.com/tree/child/leaf",
    ),
    (
        "http://example.com/tree/child/",
        "../child/leaf",
        "http://example.com/tree/child/leaf",
    ),
    (
        "http://example.com/tree/other/",
        "../../tree/child/leaf",
        "http://example.com/tree/child/leaf",
    ),
]


class TestMetamorphicRelations:
    def _tree_registry(self):
        return HIERARCHICAL.create_resource(DYNAMIC_TREE) @ Registry()

    @pytest.mark.parametrize(
        ("base_uri", "reference", "canonical"),
        EQUIVALENT_REFERENCES,
        ids=[f"{base}#{ref}" for base, ref, _ in EQUIVALENT_REFERENCES],
    )
    def test_eager_and_lazy_crawl_equivalent(
        self,
        base_uri,
        reference,
        canonical,
    ):
        """
        MR1: looking up a reference against a fully pre-crawled registry
        returns the same result as looking it up against an uncrawled one
        (which crawls on demand).
        """
        uncrawled = self._tree_registry()
        pre_crawled = uncrawled.crawl()

        eager = pre_crawled.resolver(base_uri).lookup(reference)
        lazy = uncrawled.resolver(base_uri).lookup(reference)

        assert eager.contents == lazy.contents
        assert canonical in lazy.resolver._registry
        # Discovering a subresource triggers a full crawl, after which the
        # lazy registry has converged to the eagerly crawled one.
        if canonical.endswith("/leaf"):
            assert lazy.resolver._registry == pre_crawled

    def test_dot_segments_in_base_uri_are_retained_for_fragment_refs(self):
        """
        Dot segments within the stored base URI are not collapsed: empty or
        fragment-only references use the base verbatim, while a relative
        reference containing dot segments is normalized on the way out.
        """
        registry = self._tree_registry().crawl()
        dotted_base = "http://example.com/tree/../tree/"

        # A relative reference whose own path resolves the dot segments
        # reaches the canonical document...
        assert (
            registry.resolver(dotted_base).lookup("child/leaf").contents
            == DYNAMIC_TREE["children"][0]["children"][0]
        )
        # ...but empty / fragment-only references keep the stored base URI,
        # which therefore does not match the canonical resource.
        for reference in ("", "#"):
            with pytest.raises(exceptions.Unresolvable):
                registry.resolver(dotted_base).lookup(reference)
        assert (
            registry.resolver("http://example.com/tree/")
            .lookup(
                "#",
            )
            .contents
            == DYNAMIC_TREE
        )
        # After a normalizing lookup the resolver sits on the canonical
        # base, from which a fragment-only reference succeeds.
        normalized = registry.resolver(dotted_base).lookup("child/leaf")
        assert normalized.resolver._base_uri == (
            "http://example.com/tree/child/leaf"
        )
        # ../ from a child/leaf base (leaf has no trailing slash) returns to
        # the tree root; "#" returns the leaf itself.
        assert normalized.resolver.lookup("../").contents == DYNAMIC_TREE
        assert (
            normalized.resolver.lookup("#").contents
            == (DYNAMIC_TREE["children"][0]["children"][0])
        )

    def test_lookup_of_top_level_resource_does_not_force_a_full_crawl(self):
        """
        The top-level resource is returned directly; crawling is only
        triggered when an unknown URI (e.g. a not-yet-discovered subresource)
        is requested.
        """
        uncrawled = self._tree_registry()
        resolved = uncrawled.resolver().lookup("http://example.com/tree/")
        assert resolved.resolver._registry == uncrawled
        assert "http://example.com/tree/child/leaf" not in (
            resolved.resolver._registry
        )

    def test_adding_unrelated_resources_preserves_existing_lookups(self):
        """
        MR2: a resource under an unrelated URI cannot change the resolution
        of any existing reference (neither contents nor resulting scope).
        """
        original = self._tree_registry().crawl()
        unrelated = Resource.opaque({"totally": "different"})
        extended = original.with_resource("urn:unrelated", unrelated).crawl()

        for base_uri, reference, _ in EQUIVALENT_REFERENCES:
            before = original.resolver(base_uri).lookup(reference)
            after = extended.resolver(base_uri).lookup(reference)
            assert after.contents == before.contents
            # Compare scope *URIs*; the paired registries legitimately differ
            # in how much they have so far crawled.
            assert [uri for uri, _ in after.resolver.dynamic_scope()] == [
                uri for uri, _ in before.resolver.dynamic_scope()
            ]
        # ... while the new resource is of course independently reachable.
        assert extended.contents("urn:unrelated") == {"totally": "different"}

    @pytest.mark.parametrize(
        ("base_uri", "reference", "canonical"),
        EQUIVALENT_REFERENCES,
        ids=[f"{base}#{ref}" for base, ref, _ in EQUIVALENT_REFERENCES],
    )
    def test_base_uri_normalization_is_invisible(
        self,
        base_uri,
        reference,
        canonical,
    ):
        """
        MR3: syntactically different base URIs which normalize to the same
        document (``./``, ``../``, extra path segments) resolve relative
        references to identical contents and leave the resolver on the same
        normalized base URI.
        """
        registry = self._tree_registry().crawl()
        resolved = registry.resolver(base_uri).lookup(reference)
        direct = registry.resolver().lookup(canonical)

        assert resolved.contents == direct.contents
        assert resolved.resolver._base_uri == direct.resolver._base_uri
        assert resolved.resolver._base_uri == canonical
        # Resolution is stable: the resolver now sits on the normalized URI,
        # from which a plain self-reference returns the same document.
        again = resolved.resolver.lookup(canonical)
        assert again.contents == resolved.contents
        assert again.resolver._base_uri == resolved.resolver._base_uri

    def test_combine_order_is_commutative_for_disjoint_resources(self):
        """
        MR4a: combining registries with disjoint URIs is order independent;
        eager crawling then produces identical registries.
        """
        left = HIERARCHICAL.create_resource(DYNAMIC_TREE) @ Registry()
        right = Registry().with_resource(
            "urn:right",
            Resource.opaque({"side": "right"}),
        )
        assert left.combine(right).crawl() == right.combine(left).crawl()
        combined = left.combine(right).crawl()
        assert combined.contents("urn:right") == {"side": "right"}
        assert (
            combined.resolver()
            .lookup(
                "http://example.com/tree/child/leaf",
            )
            .contents
            == DYNAMIC_TREE["children"][0]["children"][0]
        )

    def test_combine_order_only_differs_on_real_uri_conflicts(self):
        """
        MR4b: combination order has an observable effect *only* when the
        same real URI is present in both registries; the later registry's
        resource (and anchor) win.
        """
        uri = "http://example.com/shared"
        first_target = {"version": 1}
        second_target = {"version": 2}
        first = (
            HIERARCHICAL.create_resource(
                {"ID": uri, "anchors": {"a": first_target}},
            )
            @ Registry()
        ).crawl()
        second = (
            HIERARCHICAL.create_resource(
                {"ID": uri, "anchors": {"a": second_target}},
            )
            @ Registry()
        ).crawl()

        first_wins = second.combine(first).crawl()
        second_wins = first.combine(second).crawl()
        # Genuinely conflicting registries are not equal...
        assert first_wins != second_wins
        # ... and the direction of the difference is fully determined:
        assert first_wins.contents(uri) == {
            "ID": uri,
            "anchors": {"a": first_target},
        }
        assert second_wins.contents(uri) == {
            "ID": uri,
            "anchors": {"a": second_target},
        }
        assert (
            first_wins.anchor(uri, "a").value.resource.contents == first_target
        )
        assert (
            second_wins.anchor(uri, "a").value.resource.contents
            == second_target
        )
        # Combining a registry with itself short-circuits and is identical.
        assert first.combine(first) is first


# ---------------------------------------------------------------------------
# Opaque resources
# ---------------------------------------------------------------------------


class TestOpaqueSubtrees:
    def _registry(self):
        return (
            PARTIALLY_OPAQUE.create_resource(OPAQUE_TREE) @ Registry()
        ).crawl()

    def test_opaque_subtree_is_not_crawled(self):
        registry = self._registry()
        assert registry.contents("http://example.com/opaque/") == OPAQUE_TREE
        assert registry.contents("http://example.com/opaque/normal/deep") == {
            "ID": "deep",
            "anchors": {"deepAnchor": {"where": "deep"}},
        }
        # The opaque node is *not* registered under the URI its parent
        # would otherwise have derived for it -- being opaque means it
        # contributes no subresource to the graph.
        assert "http://example.com/opaque/blackbox/" not in registry
        assert "should-not-win" not in registry
        assert "http://example.com/opaque/blackbox/hidden" not in registry
        # A lookup of an anchor inside the missing opaque node surfaces a
        # missing resource before any anchor lookup is attempted.
        with pytest.raises(exceptions.NoSuchResource) as missing:
            registry.anchor(
                "http://example.com/opaque/blackbox/",
                "secret",
            )
        assert missing.value == exceptions.NoSuchResource(
            ref="http://example.com/opaque/blackbox/",
        )
        # And the root's anchors never include the opaque subtree's secret.
        assert all(name != "secret" for _uri, name in registry._anchors)

    def test_pointer_entering_opaque_node_does_not_change_base(self):
        """
        At the boundary of an opaque node, pointer walking finds the contents
        but does not treat the node as a subresource with a new base URI.
        """
        resolver = self._registry().resolver("http://example.com/opaque/")
        resolved = resolver.lookup("#/children/1")
        assert resolved.contents is OPAQUE_TREE["children"][1]
        # Base URI remains the root's; the opaque node's ``"ID"`` is ignored.
        assert resolved.resolver._base_uri == "http://example.com/opaque/"

    def test_explicitly_opaque_resource_has_no_graph_semantics(self):
        resource = Resource.opaque({"ID": "urn:ignored", "children": [{}]})
        assert resource.id() is None
        assert list(resource.subresources()) == []
        assert list(resource.anchors()) == []

        registry = Registry().with_resource("urn:opaque", resource)
        assert registry.crawl() == Registry({"urn:opaque": resource})

    def test_combine_with_opaque_resource(self):
        tree = PARTIALLY_OPAQUE.create_resource(OPAQUE_TREE) @ Registry()
        extra = Registry().with_resource(
            "urn:blob",
            Resource.opaque([1, 2, 3]),
        )
        combined = tree.combine(extra).crawl()
        assert combined.contents("urn:blob") == [1, 2, 3]
        # Crawling the opaque blob added no extra resources.
        assert "http://example.com/opaque/blackbox/hidden" not in combined


# ---------------------------------------------------------------------------
# Registry mechanics on uncrawled corpora: with_resource / @ / remove
# ---------------------------------------------------------------------------


class TestRegistryMechanics:
    def test_matmul_uses_internal_id_and_marks_uncrawled(self):
        resource = HIERARCHICAL.create_resource(DYNAMIC_TREE)
        registry = resource @ Registry()
        assert "http://example.com/tree/" in registry
        assert "http://example.com/tree/child/leaf" not in registry
        assert repr(registry) == "<Registry (1 uncrawled resource)>"
        assert len(registry.crawl()) == 3  # root, child, leaf

    def test_matmul_without_internal_id_raises(self):
        resource = PARTIALLY_OPAQUE.create_resource(OPAQUE_TREE["children"][1])
        with pytest.raises(exceptions.NoInternalID) as exc_info:
            resource @ Registry()
        assert exc_info.value.resource == resource

    def test_remove_before_and_after_crawl(self):
        registry = HIERARCHICAL.create_resource(DYNAMIC_TREE) @ Registry()
        removed = registry.remove("http://example.com/tree/")
        assert "http://example.com/tree/" not in removed
        with pytest.raises(exceptions.NoSuchResource) as exc_info:
            removed.remove("http://example.com/tree/")
        assert exc_info.value == exceptions.NoSuchResource(
            ref="http://example.com/tree/",
        )

        crawled = registry.crawl()
        without_root = crawled.remove("http://example.com/tree/")
        # Only the named URI (and its anchors) is removed; subresources are
        # independent registry entries and remain.
        assert len(without_root) == 2
        assert "http://example.com/tree/child/leaf" in without_root
        assert (
            len(without_root.remove("http://example.com/tree/child/leaf")) == 1
        )

    def test_remove_drops_anchors_for_the_uri(self):
        registry = HIERARCHICAL.create_resource(DYNAMIC_TREE) @ Registry()
        crawled = registry.crawl()
        assert (
            crawled.anchor(
                "http://example.com/tree/",
                "plainRoot",
            ).value.name
            == "plainRoot"
        )
        without_root = crawled.remove("http://example.com/tree/")
        # Both the resource and its anchor table entry are gone.
        assert ("http://example.com/tree/", "plainRoot") not in (
            without_root._anchors
        )
        with pytest.raises(exceptions.NoSuchResource):
            without_root.anchor("http://example.com/tree/", "plainRoot")
        # The child's anchors survive.
        assert (
            without_root.anchor(
                "http://example.com/tree/child/",
                "plainChild",
            ).value.name
            == "plainChild"
        )


# ---------------------------------------------------------------------------
# Retrieval callbacks: counters prove caching and laziness; no network
# ---------------------------------------------------------------------------


class CountingRetrieve:
    """
    An in-memory retriever which records every URI it was asked for.
    """

    def __init__(self, resources):
        self.resources = resources
        self.calls = []

    def __call__(self, uri):
        self.calls.append(uri)
        try:
            return self.resources[uri]
        except KeyError:
            raise exceptions.NoSuchResource(ref=uri) from None


class TestRetrieval:
    def test_retrieve_is_called_once_and_then_cached(self):
        external = Resource.opaque({"source": "callback"})
        retrieve = CountingRetrieve({"urn:external": external})
        registry = Registry(retrieve=retrieve)

        assert retrieve.calls == []  # Constructing the registry does nothing.

        first = registry.get_or_retrieve("urn:external")
        assert first.value == external
        second = first.registry.get_or_retrieve("urn:external")

        assert retrieve.calls == ["urn:external"]  # Exactly one call total.
        assert second.value is external
        assert second.registry is first.registry

    def test_crawl_is_preferred_before_retrieve(self):
        """
        A URI discoverable by crawling is found without invoking retrieval.
        """
        resource = HIERARCHICAL.create_resource(DYNAMIC_TREE)
        retrieve = CountingRetrieve({})
        registry = Registry(retrieve=retrieve).with_resource(
            "http://example.com/tree/",
            resource,
        )
        got = registry.get_or_retrieve(
            "http://example.com/tree/child/leaf",
        )
        assert (
            got.value.contents == (DYNAMIC_TREE["children"][0]["children"][0])
        )
        assert retrieve.calls == []

    def test_unknown_uri_invokes_retrieve_eagerly_on_demand(self):
        retrieve = CountingRetrieve({})
        resolver = Registry(retrieve=retrieve).resolver()
        with pytest.raises(exceptions.Unresolvable) as exc_info:
            resolver.lookup("urn:never-present")
        assert exc_info.value.ref == "urn:never-present"
        assert retrieve.calls == ["urn:never-present"]

    def test_arbitrary_retriever_error_is_unretrievable_then_unresolvable(
        self,
    ):
        def broken(uri):
            raise RuntimeError("synthetic retrieval failure")

        registry = Registry(retrieve=broken)
        with pytest.raises(exceptions.Unretrievable) as get_exc:
            registry.get_or_retrieve("urn:boom")
        assert get_exc.value == exceptions.Unretrievable(ref="urn:boom")
        assert isinstance(get_exc.value.__cause__, RuntimeError)

        with pytest.raises(exceptions.Unresolvable) as lookup_exc:
            registry.resolver().lookup("urn:boom")
        assert lookup_exc.value.ref == "urn:boom"
        assert isinstance(lookup_exc.value.__cause__, exceptions.Unretrievable)

    def test_no_such_resource_from_retriever_passes_through_without_wrapping(
        self,
    ):
        retrieve = CountingRetrieve({})
        with pytest.raises(exceptions.NoSuchResource) as exc_info:
            Registry(retrieve=retrieve).get_or_retrieve("urn:absent")
        assert exc_info.value == exceptions.NoSuchResource(ref="urn:absent")
        # Through a resolver it is reported as an unresolvable reference.
        with pytest.raises(exceptions.Unresolvable) as lookup_exc:
            Registry(retrieve=retrieve).resolver().lookup("urn:absent")
        assert lookup_exc.value == exceptions.Unresolvable(ref="urn:absent")
        assert lookup_exc.value.__cause__ is None

    def test_cannot_determine_specification_from_retriever_passes_through(
        self,
    ):
        def retrieve(uri):
            return Resource.from_contents({})  # No dialect information.

        with pytest.raises(exceptions.CannotDetermineSpecification):
            Registry(retrieve=retrieve).get_or_retrieve("urn:mystery")


# ---------------------------------------------------------------------------
# Error pinning: unhashable contents, cycles, fragments
# ---------------------------------------------------------------------------


class TestErrorContract:
    def test_resource_and_anchor_with_unhashable_contents_raise_typeerror(
        self,
    ):
        resource = Resource.opaque({"unhashable": ["contents"]})
        with pytest.raises(TypeError, match="unhashable type"):
            hash(resource)

        anchor = Anchor(name="a", resource=resource)
        with pytest.raises(TypeError, match="unhashable type"):
            hash(anchor)

        # Registry membership/lookup semantics do not rely on hashing the
        # contents and continue to work.
        registry = Registry({"urn:dict": resource})
        assert registry["urn:dict"] is resource

    @pytest.mark.parametrize(
        "build",
        [_mutual_cycle, _self_cycle],
        ids=["mutual", "self"],
    )
    def test_cyclic_subresource_structure_raises_deterministically(
        self,
        build,
    ):
        resource, repeated_uri = build()
        registry = resource @ Registry()
        with pytest.raises(exceptions.CyclicResourceStructure) as exc_info:
            registry.crawl()
        assert isinstance(exc_info.value, exceptions.Unresolvable)
        assert exc_info.value.ref == repeated_uri
        assert "cyclic" in str(exc_info.value)

        # It also deterministically aborts an on-demand lookup rather than
        # looping forever (no real clock or timeout involved).
        resolver = registry.resolver()
        with pytest.raises(exceptions.CyclicResourceStructure):
            resolver.lookup("http://example.com/cycle/a")

    def test_shared_but_acyclic_subresources_are_not_a_cycle(self):
        """
        A DAG (the same object in two sibling positions) is not a cycle and
        must remain crawlable.
        """
        shared = {"ID": "shared", "children": []}
        document = {
            "ID": "http://example.com/dag/",
            "children": [{"ID": "left/", "children": [shared]}, shared],
        }
        registry = HIERARCHICAL.create_resource(document) @ Registry()
        crawled = registry.crawl()
        assert crawled.contents("http://example.com/dag/left/shared") == shared

    def test_empty_fragment_means_the_whole_resource(self):
        registry = (
            HIERARCHICAL.create_resource(DYNAMIC_TREE) @ Registry()
        ).crawl()
        resolver = registry.resolver()
        based = registry.resolver("http://example.com/tree/")
        # An empty fragment (explicit or via a bare "#" relative ref) means
        # the whole resource rather than an anchor or pointer.
        assert (
            resolver.lookup("http://example.com/tree/").contents
            == resolver.lookup("http://example.com/tree/#").contents
            == based.lookup("#").contents
            == based.lookup("").contents
            == DYNAMIC_TREE
        )

    def test_percent_encoded_pointer_fragment(self):
        registry = HIERARCHICAL.create_resource(DYNAMIC_TREE) @ Registry()
        resolver = registry.resolver()
        # Percent-decoded pointer segment locates a key containing a space.
        root = resolver.lookup("http://example.com/tree/").resolver
        resolved = root.lookup("#/anchors/plain%20Root")
        assert resolved.contents == {"where": "spaced-root"}
        # A decoded-but-missing segment is pinned as PointerToNowhere, with
        # the ref retaining the original encoded fragment.
        with pytest.raises(exceptions.PointerToNowhere) as exc_info:
            root.lookup("#/anchors/missing%20name")
        assert exc_info.value == exceptions.PointerToNowhere(
            ref="/anchors/missing%20name",
            resource=registry.crawl()["http://example.com/tree/"],
        )

    def test_percent_encoded_plain_name_fragment_is_not_decoded_for_anchor(
        self,
    ):
        registry = HIERARCHICAL.create_resource(DYNAMIC_TREE) @ Registry()
        root = registry.crawl().resolver("http://example.com/tree/")
        # "%70lainRoot" would decode to "plainRoot", but plain-name fragment
        # matching is deliberately not percent-decoded.
        with pytest.raises(exceptions.NoSuchAnchor) as exc_info:
            root.lookup("#%70lainRoot")
        assert exc_info.value.anchor == "%70lainRoot"

        # A slash in a plain-name fragment is an invalid anchor, not a
        # pointer and not a plain anchor.
        with pytest.raises(exceptions.InvalidAnchor) as invalid:
            root.lookup("#plainRoot/extra")
        assert invalid.value.anchor == "plainRoot/extra"


# ---------------------------------------------------------------------------
# Mutation tests: deliberately breaking core invariants must make these
# invariants fail (and each is re-asserted un-mutated first).
# ---------------------------------------------------------------------------


class TestMutationsAreDetected:
    def test_reversing_dynamic_scope_order_changes_binding(self, monkeypatch):
        """
        M1: a dynamic anchor binds along the documented nearest-first
        dynamic scope; reversing that order binds an innermost target and
        must be visible as a changed result.
        """
        registry = (
            HIERARCHICAL.create_resource(DYNAMIC_TREE) @ Registry()
        ).crawl()

        def resolve_leaf():
            root = registry.resolver().lookup("http://example.com/tree/")
            child = root.resolver.lookup("#/children/0")
            leaf = child.resolver.lookup("leaf").resolver
            return leaf.lookup("#meta").contents

        root_target = {
            "ID": "http://example.com/targets/root",
            "where": "root",
        }
        child_target = {
            "ID": "http://example.com/targets/child",
            "where": "child",
        }
        assert resolve_leaf() == root_target  # Outermost dynamic anchor.

        from referencing import _core

        original_dynamic_scope = _core.Resolver.dynamic_scope

        def reversed_scope(self):
            scope = list(original_dynamic_scope(self))
            yield from reversed(scope)

        monkeypatch.setattr(_core.Resolver, "dynamic_scope", reversed_scope)

        # If the scope order were not what drives binding, reversing it
        # would leave the (root) result unchanged.
        assert resolve_leaf() == child_target

    def test_uncleared_crawl_marker_rewalks_the_graph(self, monkeypatch):
        """
        M2: after crawling, resources are marked crawled, so a second crawl
        does not re-invoke ``id_of``. Breaking the marker re-walks them.
        """
        calls = []

        def counting_id(contents):
            calls.append(1)
            return _tree_id(contents)

        specification = Specification(
            name="counting-hierarchical",
            id_of=counting_id,
            subresources_of=_tree_subresources,
            anchors_in=_tree_anchors,
            maybe_in_subresource=_in_tree,
        )
        registry = specification.create_resource(DYNAMIC_TREE) @ Registry()

        registry = registry.crawl()
        first = len(calls)
        # The root document is examined both when added (via ``id()``) and
        # when crawled; the two discovered subresources once each.
        assert first == 4
        registry = registry.crawl()
        assert len(calls) == first  # Nothing re-walked once marked crawled.

        original_crawl = Registry.crawl

        def broken_crawl(self):
            from attrs import evolve
            from rpds import HashTrieSet

            crawled = original_crawl(self)
            # Forget that anything was crawled.
            return evolve(
                crawled,
                uncrawled=HashTrieSet(crawled._resources.keys()),
            )

        monkeypatch.setattr(Registry, "crawl", broken_crawl)

        calls.clear()
        registry = registry.crawl()
        registry = registry.crawl()
        # Each re-crawl walks all three resources; with a correct marker the
        # count here would be zero.
        assert len(calls) == 6, (
            "the crawl marker mutation was not observed -- crawling did not "
            f"re-walk the graph (calls={len(calls)})"
        )

    def test_subresource_base_uri_update_is_required_for_relative_refs(
        self,
        monkeypatch,
    ):
        """
        M3: entering a subresource with a relative ID updates the base URI;
        breaking that makes the grandchild relative ref unresolvable.
        """
        registry = (
            HIERARCHICAL.create_resource(DYNAMIC_TREE) @ Registry()
        ).crawl()

        def descend():
            root = registry.resolver().lookup("http://example.com/tree/")
            child = root.resolver.lookup("#/children/0")
            return child.resolver.lookup("leaf").resolver._base_uri

        # Unmutated: the relative ``leaf`` ref resolves against the child's
        # updated base URI.
        assert descend() == "http://example.com/tree/child/leaf"

        def broken_in_subresource(self, subresource):
            # Intentionally ignore the subresource's ID: keep the old base.
            return self

        from referencing import _core

        monkeypatch.setattr(
            _core.Resolver,
            "in_subresource",
            broken_in_subresource,
        )

        # With the base update suppressed, ``leaf`` is joined against the
        # outermost base URI instead of the child's, so it points at the
        # wrong (absent) document.
        with pytest.raises(exceptions.Unresolvable) as exc_info:
            descend()
        assert exc_info.value.ref == "leaf"
        # A correctly updating implementation instead reaches the grandchild.
        monkeypatch.undo()
        assert descend() == "http://example.com/tree/child/leaf"
