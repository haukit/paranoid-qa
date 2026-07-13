"""Model-role policy: enforce that the critic is a different family from the generator.

The guardrail relies on explicitly configured families (`gen_model_family`,
`critic_model_family`) rather than parsing model-name strings, and is validated once at
graph/server startup.
"""

from __future__ import annotations

from paranoid_qa.config import settings


def validate_model_policy() -> None:
    """Raise if the generator and critic are configured on the same model family.

    A same-family critic would share the generator's failure modes, defeating the point of an
    independent check. This fails early with the conflicting values and a concrete remedy.
    """
    gen = settings.gen_model_family.strip().lower()
    critic = settings.critic_model_family.strip().lower()

    if not gen or not critic:
        raise ValueError(
            "gen_model_family and critic_model_family must both be set "
            "(PARANOID_QA_GEN_MODEL_FAMILY / PARANOID_QA_CRITIC_MODEL_FAMILY)."
        )

    if gen == critic:
        raise ValueError(
            "The verification critic must be a different model family from the generator, but both "
            f"are configured as {settings.gen_model_family!r}. Set PARANOID_QA_CRITIC_MODEL_FAMILY "
            "(and PARANOID_QA_CRITIC_MODEL) to a genuinely different family, e.g. generator "
            "'gpt-4o'/'gpt-4o-mini' with critic 'gpt-5.4'/'gpt-5.4-nano'."
        )
