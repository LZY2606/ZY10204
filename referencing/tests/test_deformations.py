"""
Deformation-centered tests for the reference graph built by a `Specification`.

These tests do not exercise any single feature in isolation. Instead they
build two small, deliberately shaped `Specification` objects along with
fixed corpora, and then assert that *deformations* which should preserve
referencing semantics in fact do:

    * eagerly crawling a registry vs. crawling lazily on demand,
    * adding an unrelated resource next to existing ones,
    * spelling the same URI with/without an empty fragment or other
      base-URI-equivalent normalization,
    * reordering `Registry.combine` arguments,

as well as pinning the dynamic behavior which such deformations must not
disturb (dynamic anchors, the dynamic scope, fallback anchors), the
caching/laziness of retrieval callbacks, and the precise exception types
raised by malformed input.

Nothing here accesses the network, the filesystem, or the wall clock; the
only "retrievers" involved are in-memory counters.
"""

from __future__ import annotations

import pytest

from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DynamicAnchor


def _hier_id_of(contents):
    if isinstance(contents, dict):
        return contents.get("ID")
    return None


def _hier_subresources_of(contents):
    if isinstance(contents, dict):
        return contents.get("children", [])
    return []


def _hier_anchors_in(
    specification,
    contents,
):
    """
    Emit same-named plain anchors before dynamic anchors, like draft 2020-12.
    """
    if not isinstance(contents, dict):
        return
    plain = contents.get("anchor")
    if plain is not None:
        yield Anchor(
            name=plain,
            resource=specification.create_resource(contents),
        )
    dynamic = contents.get("dynamic")
    if dynamic is not None:
        yield DynamicAnchor(
            name=dynamic,
            resource=specification.create_resource(contents),
        )


def _enter_every_subresource(
    *,
    segments,
    resolver,
    subresource,
):
    """
    Any pointer segment inside a child re-bases the resolver onto that child.

    The hierarchical corpus only points through its ``children`` array, so
    unlike the JSON Schema keyword-shaped rule, no keyword bookkeeping is
    needed here; every subresource reached via a pointer sets a new base URI
    (per its own ``ID``).
    """
    return resolver.in_subresource(subresource)


#: A specification with a hierarchical ``children`` array, relative ``ID``\ s
#: resolved against a parent base URI, and both plain and dynamic anchors.
HIERARCHICAL = Specification(
    name="hierarchical-deformation",
    id_of=_hier_id_of,
    subresources_of=_hier_subresources_of,
    anchors_in=_hier_anchors_in,
    maybe_in_subresource=_enter_every_subresource,
)


def _partially_opaque_id_of(contents):
    if isinstance(contents, dict) and "id" in contents:
        return contents["id"]
    return None


def _partially_opaque_subresources(contents):
    """
    Only ``visible`` children are subresources.

    Anything nested under an ``opaque`` marker is deliberately *not*
    returned: tests add those separately as `Resource.opaque` objects to
    confirm opaque documents are retrievable whole but never walked.
    """
    if isinstance(contents, dict):
        return contents.get("visible", [])
    return []


def _partially_opaque_anchors(
    specification,
    contents,
):
    if not isinstance(contents, dict):
        return
    return (
        Anchor(name=name, resource=specification.create_resource(contents))
        for name in contents.get("anchors", [])
    )


#: A specification which treats part of its graph as opaque.
PARTIALLY_OPAQUE = Specification(
    name="partially-opaque-deformation",
    id_of=_partially_opaque_id_of,
    subresources_of=_partially_opaque_subresources,
    anchors_in=_partially_opaque_anchors,
    maybe_in_subresource=_enter_every_subresource,
)


# ---------------------------------------------------------------------------
# Corpora
# ---------------------------------------------------------------------------

#: Base of the hierarchical corpus; every ID below is relative to it.
ROOT_URI = "http://example.com/hier/"

#: A three-level hierarchy. Every level defines a *dynamic* anchor ``meta``;
#: the outermost one also defines a *plain* anchor with the same name (which
#: the dynamic anchor replaces within the crawled anchor map, matching the
#: JSON Schema behavior of later anchors overriding earlier ones).
HIERARCHICAL_CONTENTS = {
    "ID": ROOT_URI,
    "dynamic": "meta",
    "marker": "root-dynamic",
    "children": [
        {
            "ID": "middle/",
            "dynamic": "meta",
            "marker": "middle-dynamic",
            "children": [
                {
                    "ID": "leaf",
                    "dynamic": "meta",
                    "marker": "leaf-dynamic",
                },
            ],
        },
    ],
}

MIDDLE_URI = "http://example.com/hier/middle/"
LEAF_URI = "http://example.com/hier/middle/leaf"

#: A separate resource reached *by URI* from within the hierarchy. It defines
#: only a plain ``meta`` anchor; a dynamic lookup from a scope containing no
#: dynamic ``meta`` therefore falls back to the reference site's own anchor
#: rather than finding this plain one dynamically.
PLAIN_META_CONTENTS = {
    "ID": "http://example.com/plain-meta",
    "anchor": "meta",
    "marker": "plain-meta-resource",
}
PLAIN_META_URI = "http://example.com/plain-meta"

#: A resource at the outermost scope which does define a dynamic ``meta``,
#: used to prove the dynamic scope can override the reference site.
OUTER_DYNAMIC_CONTENTS = {
    "ID": "http://example.com/outer-dynamic/",
    "dynamic": "meta",
    "marker": "outer-dynamic",
}
OUTER_DYNAMIC_URI = "http://example.com/outer-dynamic/"


#: The reference site for a fallback test. It defines a *dynamic* ``meta``
#: anchor; when the surrounding dynamic scope contains no matching dynamic
#: anchor (only a plain one), dynamic resolution falls back to this very
#: resource rather than to the plain anchor.
FALLBACK_SITE_CONTENTS = {
    "ID": "http://example.com/fallback-site",
    "dynamic": "meta",
    "marker": "fallback-site",
}
FALLBACK_SITE_URI = "http://example.com/fallback-site"


def hierarchical_registry(
    *,
    crawled: bool,
    retrieve=None,
) -> Registry[object]:
    """
    Build the hierarchical corpus, either eagerly or lazily crawled.
    """
    registry = (
        Registry(retrieve=retrieve) if retrieve is not None else Registry()
    )
    resources = (
        HIERARCHICAL.create_resource(HIERARCHICAL_CONTENTS),
        HIERARCHICAL.create_resource(PLAIN_META_CONTENTS),
        HIERARCHICAL.create_resource(OUTER_DYNAMIC_CONTENTS),
        HIERARCHICAL.create_resource(FALLBACK_SITE_CONTENTS),
    )
    registry = list(resources) @ registry
    return registry.crawl() if crawled else registry


#: An object hidden behind an opaque node. Its id and anchor must never be
#: discovered by crawling.
OPAQUE_HIDDEN_CONTENTS = {
    "id": "urn:deformation:hidden",
    "anchors": ["ghost"],
    "secret": 42,
}

PARTIALLY_OPAQUE_CONTENTS = {
    "id": "urn:deformation:root",
    "anchors": ["top"],
    "visible": [
        {
            "id": "urn:deformation:visible",
            "anchors": ["near"],
            "opaque": OPAQUE_HIDDEN_CONTENTS,
        },
    ],
}


def partially_opaque_registry(*, crawled: bool) -> Registry[object]:
    registry = (
        PARTIALLY_OPAQUE.create_resource(PARTIALLY_OPAQUE_CONTENTS)
        @ Registry()
    )
    if crawled:
        registry = registry.crawl()
    # The opaque document is registered explicitly at a URI of our choosing;
    # it is retrievable in full, but its internals are not walked.
    return registry.with_resource(
        "urn:deformation:opaque",
        Resource.opaque(OPAQUE_HIDDEN_CONTENTS),
    )


class CountingRetriever:
    """
    An in-memory retriever which counts how often each URI is requested.

    Instances intentionally do not cache across calls beyond what the
    registry itself does (a retrieved resource is inserted into the
    returned registry), so tests can prove registry-level laziness and
    caching without touching the network.
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

    @property
    def count(self) -> int:
        return len(self.calls)


# ---------------------------------------------------------------------------
# A reference path exercised in both the eager and on-demand variants.
# ---------------------------------------------------------------------------


def _navigate_into_leaf(resolver):
    """
    Walk from the root down into the leaf subresource.

    Returns the resolver positioned at the leaf, i.e. with the leaf as the
    current base URI and with both ancestors recorded in its dynamic scope.
    """
    root = resolver.lookup(ROOT_URI)
    assert root.contents["marker"] == "root-dynamic", (
        "the root lookup must return the root marker"
    )
    middle = root.resolver.lookup("#/children/0")
    assert middle.resolver._base_uri == MIDDLE_URI, (
        "descending a pointer into a subresource must update the base URI "
        f"to {MIDDLE_URI!r}, not {middle.resolver._base_uri!r}"
    )
    leaf = middle.resolver.lookup("leaf")
    assert leaf.resolver._base_uri == LEAF_URI
    assert leaf.contents["marker"] == "leaf-dynamic"
    return leaf.resolver


#: Every (label, reference) pair resolved against the fully navigated graph.
EQUIVALENT_LOOKUPS = [
    ("root by URI", ROOT_URI, "root-dynamic"),
    ("root by trailing empty fragment", ROOT_URI + "#", "root-dynamic"),
    ("middle by absolute URI", MIDDLE_URI, "middle-dynamic"),
    ("leaf by absolute URI", LEAF_URI, "leaf-dynamic"),
    ("plain sibling by URI", PLAIN_META_URI, "plain-meta-resource"),
]


class TestEagerVsLazyCrawl:
    def test_same_contents_for_every_lookup(self):
        """
        On-demand crawling returns the same contents as eager crawling.
        """
        eager = hierarchical_registry(crawled=True)
        lazy = hierarchical_registry(crawled=False)
        assert set(eager) == {
            ROOT_URI,
            MIDDLE_URI,
            LEAF_URI,
            PLAIN_META_URI,
            OUTER_DYNAMIC_URI,
            FALLBACK_SITE_URI,
        }
        # Before any lookup, the lazy registry has only the top-level roots.
        assert set(lazy) == {
            ROOT_URI,
            PLAIN_META_URI,
            OUTER_DYNAMIC_URI,
            FALLBACK_SITE_URI,
        }
        for label, ref, marker in EQUIVALENT_LOOKUPS:
            resolved_eager = eager.resolver(base_uri=ROOT_URI).lookup(ref)
            resolved_lazy = lazy.resolver(base_uri=ROOT_URI).lookup(ref)
            assert resolved_eager.contents == resolved_lazy.contents, label
            assert resolved_lazy.contents["marker"] == marker, label

    def test_lazy_registry_ends_up_crawled_after_demand(self):
        lazy = hierarchical_registry(crawled=False)
        leaf = _navigate_into_leaf(lazy.resolver(base_uri=ROOT_URI))
        # The resolver returned after lazy lookups carries a registry that
        # has now discovered every subresource, exactly as an eager crawl.
        assert set(leaf._registry) == set(
            hierarchical_registry(crawled=True),
        )

    def test_anchors_discovered_on_demand_match_eager(self):
        eager = hierarchical_registry(crawled=True)
        lazy = hierarchical_registry(crawled=False)
        for uri, name in [
            (ROOT_URI, "meta"),
            (MIDDLE_URI, "meta"),
            (LEAF_URI, "meta"),
            (PLAIN_META_URI, "meta"),
        ]:
            assert (
                lazy.anchor(uri, name).value == eager.anchor(uri, name).value
            ), (uri, name)


class TestUnrelatedResourceDoesNotChangeLookups:
    def test_adding_unrelated_resource_is_inert(self):
        baseline = hierarchical_registry(crawled=True)
        unrelated = Resource.opaque(contents={"totally": "unrelated"})
        extended = baseline.with_resource(
            "urn:deformation:unrelated",
            unrelated,
        )
        for _, ref, marker in EQUIVALENT_LOOKUPS:
            before = baseline.resolver(base_uri=ROOT_URI).lookup(ref)
            after = extended.resolver(base_uri=ROOT_URI).lookup(ref)
            assert before.contents == after.contents
            assert after.contents["marker"] == marker
        assert extended["urn:deformation:unrelated"] is unrelated

    def test_combining_an_unrelated_registry_is_inert(self):
        baseline = hierarchical_registry(crawled=True)
        extra = Registry().with_resource(
            "urn:deformation:elsewhere",
            Resource.opaque([1, 2, 3]),
        )
        combined = baseline.combine(extra)
        leaf_before = _navigate_into_leaf(
            baseline.resolver(base_uri=ROOT_URI),
        )
        leaf_after = _navigate_into_leaf(
            combined.resolver(base_uri=ROOT_URI),
        )
        assert leaf_before._base_uri == leaf_after._base_uri
        # The carried registries necessarily differ (one has the extra
        # resource); the deformation-preserving equivalence is the ordered
        # list of scope URIs, not the registry objects.
        scope_uris_before = [uri for uri, _ in leaf_before.dynamic_scope()]
        scope_uris_after = [uri for uri, _ in leaf_after.dynamic_scope()]
        assert scope_uris_before == scope_uris_after == [MIDDLE_URI, ROOT_URI]


class TestBaseURINormalization:
    def test_empty_fragment_is_equivalent_to_no_fragment(self):
        registry = hierarchical_registry(crawled=True)
        without = registry[ROOT_URI]
        assert registry[ROOT_URI + "#"] == without
        resolver = registry.resolver(base_uri=ROOT_URI)
        assert (
            resolver.lookup(ROOT_URI).contents
            == resolver.lookup(ROOT_URI + "#").contents
        )
        # A standalone '#' is a same-document reference to the whole
        # resource, again equivalent to the empty-fragment form.
        assert (
            resolver.lookup("#").contents == resolver.lookup(ROOT_URI).contents
        )

    def test_relative_ids_resolve_the_same_via_equivalent_bases(self):
        """
        ``a/`` and ``http://host/a`` style bases which denote the same
        document produce the same resolved URIs for relative references.
        """
        registry = hierarchical_registry(crawled=True)
        root = registry.resolver(base_uri=ROOT_URI).lookup(ROOT_URI)
        middle = root.resolver.lookup("#/children/0")
        leaf_from_absolute = middle.resolver.lookup("leaf")

        # Rebuild the exact same walk using a trailing-slash base; both must
        # converge on the canonical leaf URI.
        assert leaf_from_absolute.resolver._base_uri == LEAF_URI
        again = (
            registry.resolver(base_uri=ROOT_URI)
            .lookup(MIDDLE_URI)
            .resolver.lookup("leaf")
        )
        assert again.resolver._base_uri == LEAF_URI
        assert again.contents == leaf_from_absolute.contents

    def test_added_with_or_without_empty_fragment_collapses(self):
        resource = HIERARCHICAL.create_resource({"marker": "frag"})
        with_fragment = Registry().with_resource("urn:frag#", resource)
        without = Registry().with_resource("urn:frag", resource)
        assert with_fragment.crawl() == without.crawl()
        assert with_fragment["urn:frag"] is with_fragment["urn:frag#"]


class TestCombineEquivalence:
    def _disjoint_parts(self):
        one = hierarchical_registry(crawled=True)
        two = Registry().with_resource(
            "urn:deformation:other",
            Resource.opaque({"other": True}),
        )
        return one, two

    def test_combine_disjoint_registries_commutes(self):
        one, two = self._disjoint_parts()
        left = one.combine(two).crawl()
        right = two.combine(one).crawl()
        assert left == right

    def test_combine_preserves_all_lookups(self):
        one, two = self._disjoint_parts()
        combined = one.combine(two)
        for _, ref, marker in EQUIVALENT_LOOKUPS:
            assert (
                combined.resolver(base_uri=ROOT_URI)
                .lookup(ref)
                .contents["marker"]
                == marker
            )
        assert combined["urn:deformation:other"].contents == {"other": True}

    def test_combine_order_differs_only_on_real_uri_conflict(self):
        first_resource = PARTIALLY_OPAQUE.create_resource(
            {"id": "urn:conflict", "winner": "first"},
        )
        second_resource = PARTIALLY_OPAQUE.create_resource(
            {"id": "urn:conflict", "winner": "second"},
        )
        first = first_resource @ Registry()
        second = second_resource @ Registry()
        # No conflict -> order is irrelevant.
        disjoint_a = Registry().with_resource(
            "urn:a",
            Resource.opaque({"a": 1}),
        )
        disjoint_b = Registry().with_resource(
            "urn:b",
            Resource.opaque({"b": 2}),
        )
        assert (
            disjoint_a.combine(disjoint_b).crawl()
            == disjoint_b.combine(disjoint_a).crawl()
        )
        # Real URI conflict: the later registry wins, in a fixed order.
        assert first.combine(second).crawl()["urn:conflict"] is second_resource
        assert second.combine(first).crawl()["urn:conflict"] is first_resource


class TestPartiallyOpaqueCorpus:
    def test_opaque_subresources_are_never_crawled(self):
        eager = partially_opaque_registry(crawled=True)
        # The visible nodes are discovered...
        assert set(eager) == {
            "urn:deformation:root",
            "urn:deformation:visible",
            "urn:deformation:opaque",
        }
        # ...but the hidden id and anchor are not, in either crawl mode.
        lazy = partially_opaque_registry(crawled=False)
        for registry in (eager, lazy.crawl()):
            assert "urn:deformation:hidden" not in registry
            with pytest.raises(exceptions.NoSuchResource):
                registry["urn:deformation:hidden"]
            with pytest.raises(exceptions.NoSuchAnchor):
                registry.anchor("urn:deformation:opaque", "ghost")

    def test_opaque_document_is_retrievable_whole(self):
        registry = partially_opaque_registry(crawled=True)
        retrieved = registry["urn:deformation:opaque"]
        assert retrieved == Resource.opaque(OPAQUE_HIDDEN_CONTENTS)
        assert retrieved.contents["secret"] == 42

    def test_eager_and_lazy_agree_on_visible_lookups(self):
        eager = partially_opaque_registry(crawled=True)
        lazy = partially_opaque_registry(crawled=False)
        for ref, marker_key in [
            ("urn:deformation:root", "id"),
            ("urn:deformation:visible", "id"),
            ("urn:deformation:opaque", "secret"),
        ]:
            got_eager = eager.resolver().lookup(ref).contents
            got_lazy = lazy.resolver().lookup(ref).contents
            assert got_eager == got_lazy
            assert marker_key in got_lazy

    def test_opaque_resource_has_no_subresources_or_anchors(self):
        resource = Resource.opaque(OPAQUE_HIDDEN_CONTENTS)
        assert resource.id() is None
        assert list(resource.subresources()) == []
        assert list(resource.anchors()) == []


class TestRetrievalLazinessAndCaching:
    def _registry(self):
        remote = Resource.opaque(contents={"marker": "retrieved"})
        counter = CountingRetriever({"urn:deformation:remote": remote})
        registry = hierarchical_registry(
            crawled=False,
            retrieve=counter,
        )
        return registry, counter

    def test_retrieve_is_not_called_for_crawlable_subresources(self):
        registry, counter = self._registry()
        # The leaf is discoverable purely by crawling; the retriever must
        # never be consulted for it.
        resolver = registry.resolver(base_uri=ROOT_URI)
        leaf = _navigate_into_leaf(resolver)
        assert leaf.lookup("#").contents["marker"] == "leaf-dynamic"
        assert counter.calls == []

    def test_retrieve_called_once_then_cached_in_returned_registry(self):
        registry, counter = self._registry()
        resolver = registry.resolver()
        first = resolver.lookup("urn:deformation:remote")
        assert first.contents["marker"] == "retrieved"
        assert counter.calls == ["urn:deformation:remote"]
        # Repeated lookups reuse the resource now present in the returned
        # registry; the retriever is not invoked a second time.
        second = first.resolver.lookup("urn:deformation:remote")
        third = first.resolver.lookup("urn:deformation:remote#")
        assert second.contents == third.contents == first.contents
        assert counter.calls == ["urn:deformation:remote"]

    def test_independent_registries_each_retrieve_again(self):
        registry, counter = self._registry()
        registry.resolver().lookup("urn:deformation:remote")
        registry.resolver().lookup("urn:deformation:remote")
        # Nothing crawled back into the original immutable registry, so the
        # second independent resolver retrieves once more.
        assert counter.calls == [
            "urn:deformation:remote",
            "urn:deformation:remote",
        ]

    def test_failed_retrieval_is_not_cached(self):
        remote = Resource.opaque({})
        counter = CountingRetriever({"urn:ok": remote})
        registry = Registry(retrieve=counter)
        for _ in range(2):
            with pytest.raises(exceptions.NoSuchResource):
                registry.get_or_retrieve("urn:missing")
        assert counter.calls == ["urn:missing", "urn:missing"]


class TestDynamicAnchors:
    def test_dynamic_lookup_returns_contents_uri_and_scope(self):
        """
        A dynamic anchor is asserted via its contents, resolved URI and the
        innermost-to-outermost dynamic scope order -- not contents alone.
        """
        registry = hierarchical_registry(crawled=True)
        at_leaf = _navigate_into_leaf(registry.resolver(base_uri=ROOT_URI))

        scope = [uri for uri, _ in at_leaf.dynamic_scope()]
        assert scope == [MIDDLE_URI, ROOT_URI], scope
        # Every scope entry is paired with the same (evolved) registry.
        for _, scoped_registry in at_leaf.dynamic_scope():
            assert scoped_registry is at_leaf._registry

        resolved = at_leaf.lookup("#meta")
        assert resolved.contents["marker"] == "root-dynamic"
        # The dynamic resolution re-bases the resolver onto the resource
        # whose dynamic anchor was selected (the outermost one here).
        assert resolved.resolver._base_uri == ROOT_URI
        # The selected anchor is the one registered at the root URI, and a
        # same-document lookup from the returned resolver stays at the root.
        selected = at_leaf._registry.anchor(ROOT_URI, "meta").value
        assert isinstance(selected, DynamicAnchor)
        assert (
            resolved.resolver.lookup("#").contents
            == selected.resource.contents
        )

    def test_dynamic_scope_order_is_innermost_first(self):
        registry = hierarchical_registry(crawled=True)
        # Starting from an empty base URI, the first lookup introduces the
        # root as the first (outermost) scope entry on the *next* evolution.
        resolver = registry.resolver()
        root = resolver.lookup(ROOT_URI)
        assert [uri for uri, _ in root.resolver.dynamic_scope()] == []
        middle = root.resolver.lookup("#/children/0")
        assert [uri for uri, _ in middle.resolver.dynamic_scope()] == [
            ROOT_URI,
        ]
        leaf = middle.resolver.lookup("leaf")
        assert [uri for uri, _ in leaf.resolver.dynamic_scope()] == [
            MIDDLE_URI,
            ROOT_URI,
        ]

    def test_dynamic_anchor_falls_back_to_reference_site(self):
        """
        With only a *plain* same-named anchor in the dynamic scope, dynamic
        resolution ignores it and falls back to the reference site's own
        dynamic anchor.

        The assertion covers all three required aspects: the resolved
        contents, the resolved base URI, and the exact dynamic scope.
        """
        registry = hierarchical_registry(crawled=True)
        resolver = registry.resolver(base_uri=PLAIN_META_URI)
        at_plain = resolver.lookup(PLAIN_META_URI)
        # The current document contributes a *plain* ``meta`` anchor to the
        # registry, which dynamic resolution must not consider.
        assert isinstance(
            registry.anchor(PLAIN_META_URI, "meta").value,
            Anchor,
        )
        scope = [uri for uri, _ in at_plain.resolver.dynamic_scope()]
        assert scope == [PLAIN_META_URI], scope

        resolved = at_plain.resolver.lookup(FALLBACK_SITE_URI + "#meta")
        assert resolved.contents["marker"] == "fallback-site"
        # The resolver is re-based onto the fallback resource's own URI.
        assert resolved.resolver._base_uri == FALLBACK_SITE_URI

    def test_scope_with_only_plain_anchor_does_not_override_fallback(self):
        registry = hierarchical_registry(crawled=True)
        # Stand on the plain-meta resource (plain `meta` only), then refer
        # to the root's dynamic anchor from within a scope containing no
        # dynamic `meta`: the plain anchor must be skipped.
        resolver = registry.resolver(base_uri=PLAIN_META_URI)
        at_plain = resolver.lookup(PLAIN_META_URI)
        root_view = at_plain.resolver.lookup(ROOT_URI)
        scope = [uri for uri, _ in root_view.resolver.dynamic_scope()]
        # Re-entering the plain resource first, then moving to the root,
        # records the plain resource's URI in the dynamic scope (the
        # resolver does not de-duplicate adjacent identical entries).
        assert scope != []
        assert all(uri == PLAIN_META_URI for uri in scope)
        resolved = root_view.resolver.lookup("#meta")
        assert resolved.contents["marker"] == "root-dynamic"

    def test_outer_dynamic_anchor_overrides_reference_site(self):
        registry = hierarchical_registry(crawled=True)
        # Enter the outer dynamic resource, then navigate to the plain
        # resource while the outer resource remains in the dynamic scope.
        resolver = registry.resolver(base_uri=OUTER_DYNAMIC_URI)
        at_outer = resolver.lookup(OUTER_DYNAMIC_URI)
        at_plain = at_outer.resolver.lookup(PLAIN_META_URI)
        scope = [uri for uri, _ in at_plain.resolver.dynamic_scope()]
        assert scope, "navigating away must keep the outer scope entry"
        assert scope[-1] == OUTER_DYNAMIC_URI
        # From within the outer scope, a dynamic lookup of the root's
        # anchor is overridden by the nearer dynamic `meta` in the scope.
        resolved = at_plain.resolver.lookup(ROOT_URI + "#meta")
        assert resolved.contents["marker"] == "outer-dynamic"

    def test_same_named_plain_and_dynamic_anchor_are_both_emitted(self):
        resource = HIERARCHICAL.create_resource(
            {"anchor": "dup", "dynamic": "dup"},
        )
        anchors = list(resource.anchors())
        assert [type(each).__name__ for each in anchors] == [
            "Anchor",
            "DynamicAnchor",
        ]
        assert [each.name for each in anchors] == ["dup", "dup"]

    def test_crawl_prefers_dynamic_anchor_when_names_collide(self):
        """
        Within one resource, plain and dynamic anchors sharing a name are
        both defined, but crawling registers the dynamic one last (as in
        draft 2020-12), so same-name fragment lookups resolve dynamically.
        """
        resource = HIERARCHICAL.create_resource(
            {"ID": "urn:deformation:dup", "anchor": "x", "dynamic": "x"},
        )
        crawled = (resource @ Registry()).crawl()
        anchor = crawled.anchor("urn:deformation:dup", "x").value
        assert isinstance(anchor, DynamicAnchor)
        resolved = crawled.resolver(
            base_uri="urn:deformation:dup",
        ).lookup("urn:deformation:dup#x")
        assert resolved.contents == resource.contents


class TestPinnedExceptions:
    def test_failed_retrieve_callback_is_wrapped_as_unretrievable(self):
        def explode(uri):
            raise RuntimeError("boom")

        registry = Registry(retrieve=explode)
        with pytest.raises(exceptions.Unretrievable) as info:
            registry.get_or_retrieve("urn:deformation:kaboom")
        assert info.value.ref == "urn:deformation:kaboom"
        assert isinstance(info.value.__cause__, RuntimeError)
        # Via a resolver it is presented as an Unresolvable reference.
        with pytest.raises(exceptions.Unresolvable) as resolver_info:
            registry.resolver().lookup("urn:deformation:kaboom")
        assert resolver_info.value.ref == "urn:deformation:kaboom"

    def test_no_such_resource_callback_propagates_unwrapped(self):
        def missing(uri):
            raise exceptions.NoSuchResource(ref=uri)

        registry = Registry(retrieve=missing)
        with pytest.raises(exceptions.NoSuchResource) as info:
            registry.get_or_retrieve("urn:deformation:missing")
        assert info.value.ref == "urn:deformation:missing"

    def test_unhashable_contents_still_support_equality_but_not_hashing(self):
        unhashable = {"not": ["hashable"]}
        resource = Resource.opaque(unhashable)
        # Registries never hash contents, so the resource is fully usable.
        registry = Registry().with_resource(
            "urn:deformation:unhashable",
            resource,
        )
        assert registry["urn:deformation:unhashable"] == resource
        with pytest.raises(TypeError, match="unhashable"):
            hash(resource)
        error = exceptions.NoSuchAnchor(
            ref="urn:deformation:unhashable",
            resource=resource,
            anchor="x",
        )
        # Equality (used for assertion context) works despite unhashable
        # contents...
        assert error == exceptions.NoSuchAnchor(
            ref="urn:deformation:unhashable",
            resource=resource,
            anchor="x",
        )
        # ...but hashing the error, which would hash its resource, does not.
        with pytest.raises(TypeError, match="unhashable"):
            hash(error)

    def test_circular_subresources_raise_a_fixed_exception(self):
        one = {"ID": "urn:cycle:one"}
        two = {"ID": "urn:cycle:two", "children": [one]}
        one["children"] = [two]
        registry = HIERARCHICAL.create_resource(one) @ Registry()
        with pytest.raises(exceptions.CircularResourceStructure) as info:
            registry.crawl()
        assert info.value.uri in {
            "urn:cycle:one",
            "urn:cycle:two",
        }
        assert info.value.resource.contents in (one, two)
        assert "circular" in str(info.value).lower()
        # A demand-driven lookup surfaces the same structure error rather
        # than looping.
        with pytest.raises(exceptions.CircularResourceStructure):
            registry.resolver().lookup("urn:cycle:two")

    def test_self_referential_subresource_is_circular(self):
        node = {"ID": "urn:cycle:self"}
        node["children"] = [node]
        registry = HIERARCHICAL.create_resource(node) @ Registry()
        with pytest.raises(exceptions.CircularResourceStructure) as info:
            registry.crawl()
        assert info.value.uri == "urn:cycle:self"

    def test_circular_exception_equality_hashing_and_message(self):
        resource = Resource.opaque(("hashable", "contents"))
        error = exceptions.CircularResourceStructure(
            uri="urn:cycle:eq",
            resource=resource,
        )
        assert error == exceptions.CircularResourceStructure(
            uri="urn:cycle:eq",
            resource=resource,
        )
        assert hash(error) == hash(
            exceptions.CircularResourceStructure(
                uri="urn:cycle:eq",
                resource=resource,
            ),
        )
        assert error.__eq__(object()) is NotImplemented
        assert (error == object()) is False
        assert error != exceptions.NoSuchResource(ref="urn:cycle:eq")
        assert "urn:cycle:eq" in str(error)

    def test_shared_diamond_is_not_mistaken_for_a_cycle(self):
        leaf = {"ID": "urn:diamond:leaf", "marker": "leaf"}
        left = {"ID": "left/", "children": [leaf]}
        right = {"ID": "right/", "children": [leaf]}
        root = {"ID": "http://example.com/diamond/", "children": [left, right]}
        registry = HIERARCHICAL.create_resource(root) @ Registry()
        crawled = registry.crawl()
        # The leaf declares an absolute URN, which overrides the relative
        # path under either parent; the shared node is reached via both
        # branches but is crawled only as structure (no error), and the
        # diamond is not misreported as a cycle.
        assert crawled["urn:diamond:leaf"].contents == leaf
        assert set(crawled) == {
            "http://example.com/diamond/",
            "http://example.com/diamond/left/",
            "http://example.com/diamond/right/",
            "urn:diamond:leaf",
        }

    def test_empty_fragment_returns_the_whole_resource(self):
        registry = hierarchical_registry(crawled=True)
        assert (
            registry.resolver(base_uri=ROOT_URI)
            .lookup(ROOT_URI + "#")
            .contents
            == HIERARCHICAL_CONTENTS
        )

    def test_percent_encoded_fragment_is_not_decoded(self):
        registry = hierarchical_registry(crawled=True)
        raw_name = "meta%20space"
        with pytest.raises(exceptions.NoSuchAnchor) as info:
            registry.resolver(base_uri=ROOT_URI).lookup(
                ROOT_URI + "#" + raw_name,
            )
        # The raw, percent-encoded name is what is searched for verbatim.
        assert info.value.anchor == raw_name
        assert info.value.resource.contents == HIERARCHICAL_CONTENTS
        assert info.value.ref == ROOT_URI

    def test_pointer_like_name_without_slash_is_an_invalid_anchor(self):
        registry = hierarchical_registry(crawled=True)
        with pytest.raises(exceptions.InvalidAnchor) as info:
            registry.resolver(base_uri=ROOT_URI).lookup(
                ROOT_URI + "#children/0",
            )
        assert info.value.anchor == "children/0"


class TestRemove:
    def test_remove_drops_resource_and_its_anchors(self):
        registry = hierarchical_registry(crawled=True)
        reduced = registry.remove(MIDDLE_URI)
        assert MIDDLE_URI not in reduced
        with pytest.raises(exceptions.NoSuchResource):
            reduced[MIDDLE_URI]
        # The resource itself is gone, so asking for its anchor reports the
        # missing resource (the anchor lookup requires the resource).
        with pytest.raises(exceptions.NoSuchResource):
            reduced.anchor(MIDDLE_URI, "meta")
        # The removed anchor is also absent from the crawled anchor map.
        assert (MIDDLE_URI, "meta") not in reduced._anchors
        # Other resources and their anchors are untouched.
        assert reduced[ROOT_URI].contents == HIERARCHICAL_CONTENTS
        assert reduced.anchor(ROOT_URI, "meta").value.name == "meta"

    def test_remove_uncrawled_resource(self):
        registry = hierarchical_registry(crawled=False)
        reduced = registry.remove(PLAIN_META_URI)
        with pytest.raises(exceptions.NoSuchResource):
            reduced[PLAIN_META_URI]
        # Removing an unknown URI is fixed-shape too.
        with pytest.raises(exceptions.NoSuchResource) as info:
            reduced.remove("urn:deformation:nope")
        assert info.value.ref == "urn:deformation:nope"


# ---------------------------------------------------------------------------
# Intentional mutation tests.
#
# Each test temporarily *breaks* one piece of the machinery and asserts that
# the assertions above would fail -- proving the new tests genuinely guard
# the behavior rather than passing vacuously.
# ---------------------------------------------------------------------------


class TestIntentionalMutations:
    def test_reversing_dynamic_scope_is_caught(self, monkeypatch):
        """
        A mutation that reverses the dynamic scope must be observable.

        This proves the scope-order assertions are not vacuous: under the
        broken implementation, the innermost-first invariant fails.
        """
        registry = hierarchical_registry(crawled=True)
        at_leaf = _navigate_into_leaf(registry.resolver(base_uri=ROOT_URI))

        def scope_uris(resolver):
            return [uri for uri, _ in resolver.dynamic_scope()]

        assert scope_uris(at_leaf) == [MIDDLE_URI, ROOT_URI]

        original = type(at_leaf).dynamic_scope

        def reversed_dynamic_scope(self):
            return reversed(list(original(self)))

        monkeypatch.setattr(
            type(at_leaf),
            "dynamic_scope",
            reversed_dynamic_scope,
        )
        assert scope_uris(at_leaf) == [ROOT_URI, MIDDLE_URI]
        assert scope_uris(at_leaf) != [MIDDLE_URI, ROOT_URI], (
            "reversing the scope order must break the innermost-first "
            "invariant, otherwise the scope-order tests are vacuous"
        )

    def test_not_marking_crawl_as_done_is_caught(self, monkeypatch):
        """
        A broken crawl that forgets to clear its uncrawled marker fails the
        post-crawl invariant, so the crawl-marker tests genuinely guard it.
        """
        registry = hierarchical_registry(crawled=False)
        assert not registry.crawl()._uncrawled

        def broken_crawl(self):
            return self  # Pretend crawling happened without clearing markers.

        monkeypatch.setattr(Registry, "crawl", broken_crawl)
        assert registry.crawl()._uncrawled, (
            "a crawl which never clears its marker must fail the "
            "post-crawl invariant, otherwise the crawl-marker tests are "
            "vacuous"
        )

    def test_not_updating_subresource_base_uri_is_caught(self, monkeypatch):
        """
        Ignoring a subresource's id on pointer descent must redirect the
        base URI away from its canonical value.
        """
        registry = hierarchical_registry(crawled=True)

        def middle_base_uri_is_canonical() -> bool:
            root = registry.resolver(base_uri=ROOT_URI).lookup(ROOT_URI)
            middle = root.resolver.lookup("#/children/0")
            return middle.resolver._base_uri == MIDDLE_URI

        assert middle_base_uri_is_canonical()

        def frozen_in_subresource(self, subresource):
            return self  # Intentionally drop the subresource's base update.

        resolver_type = type(registry.resolver())
        monkeypatch.setattr(
            resolver_type,
            "in_subresource",
            frozen_in_subresource,
        )
        assert not middle_base_uri_is_canonical(), (
            "dropping the subresource base-URI update must change the "
            "resolved base URI, otherwise the base-update tests are vacuous"
        )
