import ray


@ray.remote
class MyActor:
    def __init__(self):
        self.node_id = ray.get_runtime_context().get_node_id()

    def get_node_id(self):
        return self.node_id


def test_ray_context_node_id(ray_cluster):
    actor = MyActor.remote()
    node_id = ray.get(actor.get_node_id.remote())

    assert node_id is not None
    assert len(node_id) > 0
