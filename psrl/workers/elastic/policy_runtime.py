"""
Translate runtime instance identities into the integer solver namespace.
"""

from dataclasses import replace


class PolicyIdentityMap:
    """
    Keep native gateway IDs outside the integer solver namespace.
    """

    def __init__(self):
        self._encode = {}
        self._decode = {}

    def register(self, role, model, instance_ids):
        for iid in instance_ids:
            key = (role, model, iid)
            if key in self._encode:
                continue
            solver_id = len(self._encode)
            self._encode[key] = solver_id
            self._decode[solver_id] = key

    def encode(self, role, model, iid):
        return self._encode[(role, model, iid)]

    def decode(self, solver_id):
        return self._decode[int(solver_id)]

    def signals(self, signals):
        return [replace(s, instance_id=self.encode(s.role_name, s.model_name, s.instance_id)) for s in signals]

    def decision(self, decision):
        actions = []
        for action in decision.actions:

            def entries(raw):
                return [dict(item, instance_id=self.decode(item["instance_id"])[2]) for item in raw or []]

            moves = [
                dict(
                    move,
                    source_instance_id=self.decode(move["source_instance_id"])[2],
                    destination_instance_id=self.decode(move["destination_instance_id"])[2],
                )
                for move in action.planned_request_migrations or []
            ]
            actions.append(
                replace(
                    action,
                    preferred_instance_ids=[self.decode(iid)[2] for iid in action.preferred_instance_ids or []],
                    pre_sleep_other_preferred=entries(action.pre_sleep_other_preferred),
                    pre_wake_other_preferred=entries(action.pre_wake_other_preferred),
                    planned_request_migrations=moves,
                )
            )
        return replace(decision, actions=actions)


def uses_integer_solver(policy):
    return bool(getattr(policy, "uses_integer_instance_ids", False))
