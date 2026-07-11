import pytest
from tianji.caps import Cap, Region, ResourceBudget, grant, assert_cap, disjoint


def test_cap_mint_is_unique():
    a, b = grant("tail"), grant("tail")
    assert a != b
    assert a.kind == b.kind == "tail"


def test_cap_kind_check():
    cap = grant("tail")
    assert_cap(cap, "tail")
    with pytest.raises(PermissionError):
        assert_cap(cap, "emit")


def test_region_affine_close_idempotent():
    r = Region.open("v1", 42)
    assert r.value == 42
    r.close(); r.close()
    with pytest.raises(RuntimeError):
        _ = r.value


def test_disjoint_by_rid():
    a = Region.open("a", 1); b = Region.open("b", 2)
    assert disjoint(a, b)
    assert not disjoint(a, a)


def test_budget_allocate_and_free():
    b = ResourceBudget("vram", 1000)
    b.allocate(400); assert b.used_bytes == 400; assert b.remaining == 600
    b.free(200); assert b.used_bytes == 200
    with pytest.raises(MemoryError):
        b.allocate(1000)


def test_budget_no_negative():
    b = ResourceBudget("vram", 100)
    with pytest.raises(ValueError):
        b.allocate(-1)
    b.free(50)
    assert b.used_bytes == 0
