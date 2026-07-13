from paranoid_qa.workflow.routing import Route


def test_route_accepts_valid_kinds():
    assert Route(kind="specific").kind == "specific"
    assert Route(kind="aggregate").kind == "aggregate"
