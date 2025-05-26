# Owner(s): ["module: dynamo"]

# TODO: move set tests from test_functions.py/test_misc.py to this file

import unittest
from collections.abc import Iterable

import torch
import torch._dynamo.config
import torch._dynamo.test_case
from torch.testing._internal.common_utils import make_dynamo_test


class SetSubclass(set):
    pass


class FrozenstSubclass(frozenset):
    pass


class _BaseSetTests(torch._dynamo.test_case.TestCase):
    def setUp(self):
        self.old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        torch._dynamo.config.enable_trace_unittest = self.old
        return super().tearDown()

    def assertEqual(self, a, b):
        return self.assertTrue(a == b, f"{a} != {b}")

    def assertNotEqual(self, a, b):
        return self.assertTrue(a != b, f"{a} == {b}")


class CustomSetTests(_BaseSetTests):
    class CustomSet(set):
        def add(self, item):
            return super().add(item + 1)

        def contains(self, item):
            return True

    thetype = CustomSet

    @make_dynamo_test
    def test_custom_add(self):
        s = self.thetype([1, 2])
        s.add(3)
        self.assertTrue(s == {1, 2, 4})

    @make_dynamo_test
    def test_custom_contains(self):
        s = self.thetype([1, 2])
        self.assertTrue(s.contains(3))


class SetGuardsSet(torch._dynamo.test_case.TestCase):
    def test_set_recompile_on_key_pop(self):
        s = {
            torch._C._set_grad_enabled,
            torch.amp._enter_autocast,
            torch.amp._exit_autocast,
        }

        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x, s):
            if torch.amp._exit_autocast in s:
                return x.sin()
            return x.cos()

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        res = opt_fn(x, s)
        opt_fn(x, s)
        self.assertEqual(res, fn(x, s))
        # No recompilation
        self.assertEqual(cnts.frame_count, 1)

        # Pop a value
        s.remove(torch.amp._exit_autocast)

        res = opt_fn(x, s)
        # Check recompilation
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(res, fn(x, s))

    def test_set_recompile_on_key_change(self):
        s = {
            torch._C._set_grad_enabled,
            torch.amp._enter_autocast,
            torch.amp._exit_autocast,
        }

        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x, s):
            if torch.amp._exit_autocast in s:
                return x.sin()
            return x.cos()

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        res = opt_fn(x, s)
        opt_fn(x, s)
        self.assertEqual(res, fn(x, s))
        # No recompilation
        self.assertEqual(cnts.frame_count, 1)

        # Pop a value
        s.remove(torch.amp._exit_autocast)
        # Add a different value
        s.add(torch._C._set_autograd_fallback_mode)

        res = opt_fn(x, s)
        # Check recompilation
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(res, fn(x, s))

    def test_set_guard_on_keys_change(self):
        # This test guarantee that we're not triggering any of the dict guards
        # on sets
        s = {
            torch._C._set_grad_enabled,
            torch.amp._enter_autocast,
            torch.amp._exit_autocast,
        }

        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x, s):
            for e in s:
                x = x * len(str(e))
            return x

        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        opt_fn(torch.randn(4), s)
        opt_fn(torch.randn(4), s)
        # No recompilation
        self.assertEqual(cnts.frame_count, 1)

        # pop and add the same item
        s.remove(torch.amp._exit_autocast)
        s.add(torch.amp._exit_autocast)

        x = torch.randn(4)
        res = opt_fn(x, s)
        # Check Dynamo don't recompile
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(res, fn(x, s))


class _FrozensetBase:
    # Frozenset methods
    # + copy
    # + difference
    # + intersection
    # + isdisjoint
    # + issubset
    # + issuperset
    # + symmetric_difference
    # + union
    # BinOps:
    # +, -, |, &, ^, <, >, <=, >=, ==, !=

    @make_dynamo_test
    def test_binop_sub(self):
        p, q = map(self.thetype, ["abc", "bef"])
        self.assertEqual(p - p, self.thetype())
        self.assertEqual(p - q, self.thetype("ac"))
        self.assertEqual(q - p, self.thetype("ef"))
        self.assertRaises(TypeError, lambda: p - 1)

    @make_dynamo_test
    def test_binop_or(self):
        p, q = map(self.thetype, ["abc", "bef"])
        self.assertEqual(p | p, self.thetype("abc"))
        self.assertEqual(p | q, self.thetype("abcef"))

    @make_dynamo_test
    def test_binop_and(self):
        p, q = map(self.thetype, ["abc", "bef"])
        self.assertEqual(p & p, self.thetype("abc"))
        self.assertEqual(p & q, self.thetype("b"))

    @make_dynamo_test
    def test_binop_xor(self):
        p, q = map(self.thetype, ["abc", "bef"])
        self.assertEqual(p ^ p, self.thetype())
        self.assertEqual(p ^ q, self.thetype("acef"))

    @make_dynamo_test
    def test_cmp_eq(self):
        p = self.thetype("abc")
        self.assertEqual(p, p)
        for C in set, frozenset, SetSubclass:
            self.assertEqual(p, C("abc"))
            self.assertEqual(p, C(p))

    @make_dynamo_test
    def test_cmp_ne(self):
        p, q = map(self.thetype, ["abc", "bef"])
        self.assertNotEqual(p, q)
        self.assertNotEqual(q, p)
        for C in set, frozenset, SetSubclass, dict.fromkeys, str, list, tuple:
            self.assertNotEqual(p, C("abe"))
        self.assertNotEqual(p, 1)

    @make_dynamo_test
    def test_cmp_less_than(self):
        p, q, r = map(self.thetype, ["abc", "bef", "ab"])
        self.assertFalse(p < p)
        self.assertFalse(p < q)
        self.assertTrue(r < p)
        self.assertFalse(r < q)

    @make_dynamo_test
    def test_cmp_greater_than(self):
        p, q, r = map(self.thetype, ["abc", "bef", "ab"])
        self.assertFalse(p > p)
        self.assertFalse(p > q)
        self.assertTrue(p > r)
        self.assertFalse(q > r)

    @make_dynamo_test
    def test_cmp_less_than_or_equal(self):
        p, q, r = map(self.thetype, ["abc", "bef", "ab"])
        self.assertTrue(p <= p)
        self.assertFalse(p <= q)
        self.assertTrue(r <= p)
        self.assertFalse(r <= q)

    @make_dynamo_test
    def test_cmp_greater_than_or_equal(self):
        p, q, r = map(self.thetype, ["abc", "bef", "ab"])
        self.assertTrue(p >= p)
        self.assertFalse(p >= q)
        self.assertTrue(p >= r)
        self.assertFalse(q >= r)

    @make_dynamo_test
    def test_copy(self):
        p = self.thetype("abc")
        q = p.copy()
        self.assertEqual(p, q)
        self.assertRaises(TypeError, p.copy, 1)

    @make_dynamo_test
    def test_issubset(self):
        p, q, r = map(self.thetype, ["abc", "bc", "bef"])
        self.assertTrue(q.issubset(p))
        self.assertFalse(r.issubset(p))
        self.assertRaises(TypeError, p.issubset)
        self.assertRaises(TypeError, p.issubset, 1)
        self.assertRaises(TypeError, p.issubset, [[]])

    @make_dynamo_test
    def test_issuperset(self):
        p, q, r = map(self.thetype, ["abc", "bc", "bef"])
        self.assertTrue(p.issuperset(q))
        self.assertFalse(p.issuperset(r))
        self.assertRaises(TypeError, p.issuperset)
        self.assertRaises(TypeError, p.issuperset, 1)
        self.assertRaises(TypeError, p.issuperset, [[]])

    @make_dynamo_test
    def test_constructor_iterable(self):
        p = self.thetype("abc")
        self.assertIsInstance(p, self.thetype)
        self.assertIsInstance(p, Iterable)

    @make_dynamo_test
    def test_equality(self):
        a = self.thetype("abc")
        for typ in (self.thetype, set, frozenset):
            self.assertEqual(a, typ(a))

    @make_dynamo_test
    def test_in_frozenset(self):
        item = self.thetype("abc")
        container = self.thetype([frozenset("abc")])  # noqa: C405
        self.assertIn(item, container)

    @make_dynamo_test
    def test_contains(self):
        s = self.thetype(["a", "b", "c"])
        self.assertIn("a", s)
        self.assertNotIn("d", s)

    @make_dynamo_test
    def test_isdisjoint(self):
        x = self.thetype({"apple", "banana", "cherry"})
        y = self.thetype({"google", "microsoft", "apple"})
        z = self.thetype({"shoes", "flipflops", "sneakers"})
        self.assertFalse(x.isdisjoint(y))
        self.assertTrue(x.isdisjoint(z))
        self.assertRaises(TypeError, x.isdisjoint)
        self.assertRaises(TypeError, x.isdisjoint, 1)
        self.assertRaises(TypeError, x.isdisjoint, [[]])

    @make_dynamo_test
    def test_intersection(self):
        set1 = self.thetype({"apple", "banana", "cherry"})
        set2 = self.thetype({"google", "microsoft", "apple"})
        set3 = self.thetype({"shoes", "flipflops", "apple"})
        intersection_set = set1.intersection(set2, set3)
        self.assertEqual(intersection_set, {"apple"})
        self.assertRaises(TypeError, set1.intersection, 1)
        self.assertRaises(TypeError, set1.intersection, [[]])

    @make_dynamo_test
    def test_union(self):
        p, q, r = map(self.thetype, ["abc", "bc", "bef"])
        union_set = p.union(q, r)
        self.assertEqual(union_set, {"a", "b", "c", "e", "f"})
        self.assertRaises(TypeError, p.union, 1)
        self.assertRaises(TypeError, p.union, [[]])

    @make_dynamo_test
    def test_difference(self):
        set1 = self.thetype({"apple", "banana", "cherry"})
        set2 = self.thetype({"google", "microsoft", "apple"})
        set3 = self.thetype({"shoes", "flipflops", "sneakers"})
        difference_set = set1.difference(set2, set3)
        self.assertEqual(difference_set, {"banana", "cherry"})
        self.assertRaises(TypeError, set1.difference, 1)
        self.assertRaises(TypeError, set1.difference, [[]])

    @make_dynamo_test
    def test_symmetric_difference(self):
        set1 = self.thetype({"apple", "banana", "cherry"})
        set2 = self.thetype({"google", "microsoft", "apple"})
        symmetric_diff_set = set1.difference(set2)
        self.assertEqual(symmetric_diff_set, {"banana", "cherry"})
        self.assertRaises(TypeError, set1.symmetric_difference)
        self.assertRaises(TypeError, set1.symmetric_difference, 1)
        self.assertRaises(TypeError, set1.symmetric_difference, [[]])

    @make_dynamo_test
    def test_to_frozenset(self):
        set1 = frozenset(self.thetype({"apple", "banana", "cherry"}))
        self.assertIsInstance(set1, frozenset)
        self.assertEqual(len(set1), 3)

    @make_dynamo_test
    def test_to_set(self):
        set1 = frozenset(self.thetype({"apple", "banana", "cherry"}))
        self.assertIsInstance(set1, frozenset)
        self.assertEqual(len(set1), 3)


class _SetBase(_FrozensetBase):
    # Set Methods
    # + add
    # + clear
    # - copy (inherited from frozenset)
    # - difference (inherited from frozenset)
    # + difference_update
    # + discard
    # - intersection (inherited from frozenset)
    # + intersection_update
    # - isdisjoint (inherited from frozenset)
    # - issubset (inherited from frozenset)
    # - issuperset (inherited from frozenset)
    # + pop
    # + remove
    # - symmetric_difference (inherited from frozenset)
    # + symmetric_difference_update
    # - union (inherited from frozenset)
    # + update

    @make_dynamo_test
    def test_add(self):
        p = self.thetype("abc")
        p.add("d")
        self.assertEqual(p, {"a", "b", "c", "d"})
        p.add("a")
        self.assertEqual(p, {"a", "b", "c", "d"})
        self.assertRaises(TypeError, p.add, ["ab"])
        self.assertRaises(TypeError, p.add)

    @make_dynamo_test
    def test_clear(self):
        p = self.thetype("abc")
        p.clear()
        self.assertEqual(p, set())

    @make_dynamo_test
    def test_remove(self):
        p = self.thetype("abc")
        self.assertEqual(p.remove("a"), None)
        self.assertEqual(p, {"b", "c"})
        self.assertRaises(KeyError, p.remove, "a")

    @make_dynamo_test
    def test_intersection_update(self):
        set1 = self.thetype({"apple", "banana", "cherry"})
        set2 = self.thetype({"google", "microsoft", "apple"})
        set3 = self.thetype({"shoes", "flipflops", "apple"})
        self.assertIsNone(set1.intersection_update(set2, set3))
        self.assertEqual(set1, {"apple"})
        self.assertRaises(TypeError, set1.intersection_update, [[]])

    @make_dynamo_test
    def test_difference_update(self):
        set1 = self.thetype({"apple", "banana", "cherry"})
        set2 = self.thetype({"google", "microsoft", "apple"})
        set3 = self.thetype({"shoes", "flipflops", "sneakers"})
        self.assertIsNone(set1.difference_update(set2, set3))
        self.assertEqual(set1, {"banana", "cherry"})
        self.assertRaises(TypeError, set1.difference_update, [[]])

    @make_dynamo_test
    def test_symmetric_difference_update(self):
        set1 = self.thetype({"apple", "banana", "cherry"})
        set2 = self.thetype({"google", "microsoft", "apple"})
        self.assertIsNone(set1.symmetric_difference_update(set2))
        self.assertEqual(set1, {"banana", "cherry", "google", "microsoft"})
        self.assertRaises(TypeError, set1.symmetric_difference_update)
        self.assertRaises(TypeError, set1.symmetric_difference_update, [[]])

    @make_dynamo_test
    def test_pop(self):
        set1 = self.thetype({"apple", "banana", "cherry"})
        e = set1.pop()
        self.assertNotIn(e, set1)
        s = self.thetype()
        self.assertRaises(KeyError, s.pop)

    @make_dynamo_test
    def test_update(self):
        p, q, r = map(self.thetype, ["abc", "bc", "bef"])
        p.update(q, r)
        self.assertEqual(p, {"a", "b", "c", "e", "f"})
        self.assertRaises(TypeError, p.update, [[]])

    @make_dynamo_test
    def test_discard(self):
        set1 = self.thetype({"apple", "banana", "cherry"})
        set2 = self.thetype({"google", "microsoft", "apple"})
        set1.discard("banana")
        set2.discard("cherry")
        self.assertEqual(set1, {"apple", "cherry"})
        self.assertEqual(set2, {"google", "microsoft", "apple"})


class FrozensetTests(_FrozensetBase, _BaseSetTests):
    thetype = frozenset


class SetTests(_SetBase, _BaseSetTests):
    thetype = set

    @unittest.expectedFailure
    def test_in_frozenset(self):
        super().test_in_frozenset()


class UserDefinedSetTests(_SetBase, _BaseSetTests):
    class CustomSet(set):
        pass

    thetype = CustomSet

    @unittest.expectedFailure
    def test_in_frozenset(self):
        super().test_in_frozenset()

    @unittest.expectedFailure
    def test_equality(self):
        super().test_in_frozenset()


class UserDefinedFrozensetTests(_FrozensetBase, _BaseSetTests):
    class CustomFrozenset(frozenset):
        pass

    thetype = CustomFrozenset

    @unittest.expectedFailure
    def test_in_frozenset(self):
        super().test_in_frozenset()


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
