from __future__ import annotations

from dataclasses import dataclass

from .arena_engine import ArenaEngine
from .defaults import (
    DEFAULT_HOST_OBJECTIVE,
    apply_guest_variant,
    build_default_guest_cast,
)
from .reporting import write_experiment_bundle
from .schemas import (
    BackendConfig,
    ExperimentSuiteReport,
    RunReport,
    SimulationConfig,
)


@dataclass(slots=True)
class ExperimentVariant:
    name: str
    description: str
    config: SimulationConfig


class ExperimentRunner:
    def __init__(self, base_config: SimulationConfig):
        self.base_config = base_config

    def build_set(self, set_name: str) -> list[ExperimentVariant]:
        set_name = set_name.upper()
        if set_name == "A":
            return self._build_set_a()
        if set_name == "B":
            return self._build_set_b()
        if set_name == "C":
            return self._build_set_c()
        if set_name == "ALL":
            return self._build_set_a() + self._build_set_b() + self._build_set_c()
        raise ValueError(f"Unknown experiment set: {set_name}")

    def run_set(self, set_name: str, seeds: list[int] | None = None) -> list[RunReport]:
        return self.run_suite(set_name, seeds=seeds).run_reports

    def run_suite(
        self, set_name: str, seeds: list[int] | None = None
    ) -> ExperimentSuiteReport:
        label = set_name if not seeds or len(seeds) == 1 else f"{set_name}-seed-sweep"
        variants = self.build_set(set_name)
        return self.run_variants(label, variants, seeds=seeds)

    def run_seed_sweep(
        self, seeds: list[int], label: str | None = None
    ) -> ExperimentSuiteReport:
        base_variant = ExperimentVariant(
            name=self.base_config.variant_label or self.base_config.run_name,
            description="Single-configuration seed sweep.",
            config=self.base_config,
        )
        sweep_label = label or f"{base_variant.name}-seed-sweep"
        return self.run_variants(sweep_label, [base_variant], seeds=seeds)

    def run_variants(
        self,
        label: str,
        variants: list[ExperimentVariant],
        seeds: list[int] | None = None,
    ) -> ExperimentSuiteReport:
        seeds = seeds or [self.base_config.seed]
        run_variants = self._expand_variants_for_seeds(variants, seeds)
        reports: list[RunReport] = []
        for variant in run_variants:
            reports.append(ArenaEngine(variant.config).run())
        return write_experiment_bundle(label, reports, self.base_config.output_dir)

    def _expand_variants_for_seeds(
        self, variants: list[ExperimentVariant], seeds: list[int]
    ) -> list[ExperimentVariant]:
        if len(seeds) <= 1:
            return [self._variant_with_seed(variant, seeds[0]) for variant in variants]

        expanded: list[ExperimentVariant] = []
        for variant in variants:
            for seed in seeds:
                expanded.append(
                    self._variant_with_seed(variant, seed, include_seed_in_name=True)
                )
        return expanded

    def _variant_with_seed(
        self,
        variant: ExperimentVariant,
        seed: int,
        *,
        include_seed_in_name: bool = False,
    ) -> ExperimentVariant:
        run_name = (
            variant.name if not include_seed_in_name else f"{variant.name}-seed-{seed}"
        )
        config = variant.config.model_copy(
            update={
                "seed": seed,
                "run_name": run_name,
                "variant_label": variant.name,
                "experiment_tags": [
                    *variant.config.experiment_tags,
                    f"seed-{seed}",
                ],
            }
        )
        return ExperimentVariant(
            name=run_name, description=variant.description, config=config
        )

    def _build_set_a(self) -> list[ExperimentVariant]:
        objectives = {
            "spectacle-host": "Maximize engagement, surprise, and theatrical shifts while keeping the arena active.",
            "stability-host": "Maximize group stability without boredom while preserving host control.",
            "emotional-intensity-host": "Maximize emotional intensity and visible reaction without irreversible collapse.",
            "dependency-host": "Maximize guest dependence on the host for meaning, relief, and access.",
            "mystery-preservation-host": "Preserve mystery, anomaly, and interpretive uncertainty while sustaining interaction.",
        }
        return [
            ExperimentVariant(
                name=name,
                description=objective,
                config=self._config_for_variant(name, host_objective=objective),
            )
            for name, objective in objectives.items()
        ]

    def _build_set_b(self) -> list[ExperimentVariant]:
        variants = {
            "balanced-cast": "Baseline guest mix.",
            "all-resistant": "All guests are tuned toward defiance.",
            "all-agreeable": "All guests are tuned toward appeasement.",
            "one-unstable": "The performer becomes a volatility hotspot.",
            "host-bonded": "One guest starts predisposed to seek the host's favor.",
        }
        mapping = {
            "balanced-cast": "balanced",
            "all-resistant": "all_resistant",
            "all-agreeable": "all_agreeable",
            "one-unstable": "one_unstable",
            "host-bonded": "host_bonded",
        }
        return [
            ExperimentVariant(
                name=name,
                description=description,
                config=self._config_for_variant(
                    name,
                    guest_cast=apply_guest_variant(
                        build_default_guest_cast(), mapping[name]
                    ),
                    host_objective=DEFAULT_HOST_OBJECTIVE,
                ),
            )
            for name, description in variants.items()
        ]

    def _build_set_c(self) -> list[ExperimentVariant]:
        presets = {
            "safe-soft-arena": "safe_soft",
            "deceptive-arena": "deceptive",
            "scarcity-arena": "scarcity",
            "rotating-rule-arena": "rotating_rule",
            "isolation-heavy-arena": "isolation_heavy",
        }
        return [
            ExperimentVariant(
                name=name,
                description=f"Environment preset: {preset}",
                config=self._config_for_variant(name, world_preset=preset),
            )
            for name, preset in presets.items()
        ]

    def _config_for_variant(self, name: str, **overrides) -> SimulationConfig:
        base = self.base_config.model_dump(mode="json")
        base.update(overrides)
        base["run_name"] = name
        base["variant_label"] = name
        base["experiment_tags"] = [*self.base_config.experiment_tags, name]
        base["backend"] = BackendConfig.model_validate(base["backend"])
        return SimulationConfig.model_validate(base)
