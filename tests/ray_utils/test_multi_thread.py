import ray


@ray.remote
class ThreadedActor:
    def task_1(self):
        return "task_1_done"

    def task_2(self):
        return "task_2_done"


def test_multi_thread_concurrent(ray_cluster):
    a = ThreadedActor.options(max_concurrency=2).remote()
    results = ray.get([a.task_1.remote(), a.task_2.remote()])

    assert "task_1_done" in results
    assert "task_2_done" in results
