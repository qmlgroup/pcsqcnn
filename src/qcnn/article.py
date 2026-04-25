"""Article-alignment constants and inspection helpers for ``qcnn``.

The module centralizes the article-aligned defaults used across normalized data
preparation and encoding:

- ``ARTICLE_ALIGNED_IMAGE_SIZE = 16``
- ``ARTICLE_ALIGNED_BRIGHTNESS_RANGE = (0.0, math.pi)``
- ``ARTICLE_ALIGNED_SAMPLES_PER_CLASS = 1000``

It also exposes the documented FRQI mismatch message used by the encoder:
``FrqiEncoder2D`` intentionally omits the article's global ``1 / sqrt(XY)``
normalization factor, so encoded sample norms are ``sqrt(XY)`` rather than
``1``.

The helpers in this module do not mutate model state. They compute how a chosen
configuration differs from the current article-aligned defaults. Runtime
warnings for parameter-value deviations are intentionally suppressed so local
experiments can vary those values without noisy output.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence

ARTICLE_ALIGNED_IMAGE_SIZE = 16
ARTICLE_ALIGNED_BRIGHTNESS_RANGE = (0.0, math.pi)
ARTICLE_ALIGNED_SAMPLES_PER_CLASS = 1000

ENCODER_NORMALIZATION_MISMATCH = (
    "FrqiEncoder2D intentionally omits the article's global 1/sqrt(XY) "
    "normalization factor. Encoded sample norms are sqrt(XY), not 1."
)


def article_alignment_warnings(
    *,
    image_size: int | None = None,
    brightness_range: Sequence[float] | None = None,
    samples_per_class: int | None = None,
    scaled_image_size: int | None = None,
    max_offset: int | None = None,
) -> tuple[str, ...]:
    """Return human-readable warnings for article-alignment mismatches.

    Args:
        image_size: Optional image side length to compare against the
            article-aligned default ``16``.
        brightness_range: Optional encoder-side brightness interval ``(a, b)``
            to compare against the article-aligned default ``(0.0, math.pi)``.
        samples_per_class: Optional number of training examples per class to
            compare against the article-aligned translated-benchmark default
            ``1000``.
        scaled_image_size: Optional intermediate resize side length used before
            placing digits onto the final ``image_size x image_size`` canvas.
        max_offset: Optional maximum integer translation radius used when
            placing resized digits on the final canvas.

    Returns:
        A tuple of warning messages. The tuple is empty when all provided values
        match the article-aligned defaults.

    Notes:
        The helper is pure: it only builds messages and never emits warnings on
        its own. ``warn_for_article_alignment`` returns the same alignment
        messages for programmatic inspection, but does not emit them at runtime.
    """

    messages: list[str] = []

    if image_size is not None and image_size != ARTICLE_ALIGNED_IMAGE_SIZE:
        messages.append(
            "image_size deviates from the article-aligned default of "
            f"{ARTICLE_ALIGNED_IMAGE_SIZE}."
        )

    if brightness_range is not None:
        normalized_range = (float(brightness_range[0]), float(brightness_range[1]))
        if normalized_range != ARTICLE_ALIGNED_BRIGHTNESS_RANGE:
            messages.append(
                "brightness_range deviates from the article-aligned default of "
                f"{ARTICLE_ALIGNED_BRIGHTNESS_RANGE}."
            )

    if samples_per_class is not None and samples_per_class != ARTICLE_ALIGNED_SAMPLES_PER_CLASS:
        messages.append(
            "samples_per_class deviates from the article-aligned translated-benchmark default of "
            f"{ARTICLE_ALIGNED_SAMPLES_PER_CLASS}."
        )

    translation_enabled = False
    if max_offset is not None and max_offset != 0:
        translation_enabled = True
    if (
        scaled_image_size is not None
        and image_size is not None
        and scaled_image_size != image_size
    ):
        translation_enabled = True
    if translation_enabled:
        messages.append(
            "translation preprocessing deviates from the article-aligned centered "
            "resize-only pipeline."
        )

    return tuple(messages)


def warn_for_article_alignment(
    *,
    image_size: int | None = None,
    brightness_range: Sequence[float] | None = None,
    samples_per_class: int | None = None,
    scaled_image_size: int | None = None,
    max_offset: int | None = None,
    include_encoder_mismatch: bool = False,
    stacklevel: int = 2,
) -> tuple[str, ...]:
    """Return documented deviations from article defaults.

    Args:
        image_size: Optional image side length to compare against the
            article-aligned default.
        brightness_range: Optional encoder-side brightness interval to compare
            against the article-aligned default.
        samples_per_class: Optional training-subset size per class to compare
            against the article-aligned default.
        scaled_image_size: Optional intermediate resize side length used before
            placing digits onto the final ``image_size x image_size`` canvas.
        max_offset: Optional maximum integer translation radius used when
            placing resized digits on the final canvas.
        include_encoder_mismatch: Whether to also emit the documented FRQI
            normalization mismatch warning for ``FrqiEncoder2D``.
        stacklevel: ``warnings.warn`` stacklevel propagated to every emitted
            warning.

    Returns:
        The tuple of article-alignment messages plus the optional encoder
        mismatch message when requested.

    Notes:
        Parameter-value deviations from article defaults are no longer emitted
        as runtime warnings. The only remaining emitted warning is the optional
        encoder-normalization mismatch, because it documents a behavioral
        convention of the FRQI-like encoder rather than a chosen hyperparameter.
    """

    messages = list(
        article_alignment_warnings(
            image_size=image_size,
            brightness_range=brightness_range,
            samples_per_class=samples_per_class,
            scaled_image_size=scaled_image_size,
            max_offset=max_offset,
        )
    )

    if include_encoder_mismatch:
        messages.append(ENCODER_NORMALIZATION_MISMATCH)
        warnings.warn(ENCODER_NORMALIZATION_MISMATCH, UserWarning, stacklevel=stacklevel)

    return tuple(messages)
