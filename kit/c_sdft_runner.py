"""Importable Ray runner applying only C's registered context allowances."""
from kit import sdft_verl as v
from kit.c_sdft_entry import legacy_config


class CTaskRunner(v.SdftTaskRunner):
    def run(self, config):
        from kit import sdft_objective as obj, v4_contract as contract
        original_obj = obj.check_trainer_config; original_contract = contract.check_resolved
        obj.check_trainer_config = lambda c: original_obj(legacy_config(c))
        contract.check_resolved = lambda c, *a, **kw: original_contract(legacy_config(c), *a, **kw)
        try: return super().run(config)
        finally:
            obj.check_trainer_config = original_obj; contract.check_resolved = original_contract
