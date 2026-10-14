import ray
from ray import get_runtime_context


# Define a simple Actor
@ray.remote
class MyActor:
    def __init__(self):
        self.value = 42

    def get_val(self):
        return self.value

    def say_hello(self):
        self.value += 1
        return "Hello from MyActor!"

    def get_current_actor_handle(self):
        return get_runtime_context().current_actor


def test_get_handle(ray_cluster):
    actor_handle = MyActor.remote()
    another_actor_handle = MyActor.remote()

    actor_handle_outside = ray.get(actor_handle.get_current_actor_handle.remote())
    assert actor_handle_outside is not None

    result = ray.get(actor_handle_outside.say_hello.remote())
    assert result == "Hello from MyActor!"

    result_direct = ray.get(actor_handle.say_hello.remote())
    assert result_direct == "Hello from MyActor!"

    val = ray.get(actor_handle.get_val.remote())
    assert val == 44  # incremented twice
    another_val = ray.get(another_actor_handle.get_val.remote())
    assert another_val == 42
