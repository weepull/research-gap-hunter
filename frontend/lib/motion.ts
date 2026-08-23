/**
 * Motion system.
 *
 * Apple deliberately replaced the physics triplet (mass/stiffness/damping)
 * with two designer-facing parameters — **damping ratio** (how much overshoot)
 * and **response** (how quickly the value reaches the target, in seconds).
 * Motion's spring API exposes `bounce` + `duration`, which maps onto those
 * directly: `bounce: 0` is critically damped (damping 1.0), and `duration` is
 * response, not a fixed runtime — a spring has no fixed runtime, its settle
 * time emerges from the parameters.
 *
 * Springs rather than CSS transitions because everything a user can touch has
 * to be interruptible: a spring animates from the *presentation* value and
 * carries velocity through a re-target, so a gesture can grab a moving element
 * mid-flight and reverse it without a jump or a velocity discontinuity.
 */

import { useReducedMotion } from "motion/react";
import { useCallback, useMemo, useRef } from "react";

export type Spring = {
  type: "spring";
  bounce: number;
  duration: number;
};

/**
 * The house springs.
 *
 * Critically damped is the default. Overshoot only appears in `sheet` and
 * `fling` — the two presets reached *after* a gesture carried momentum. Bounce
 * on a menu that merely faded in feels wrong; bounce on a card you threw feels
 * right, and that distinction is the whole rule.
 */
export const SPRING = {
  /** Pointer-down feedback. Fastest thing in the system — response is what
   *  latency feels like, and press feedback is the one place it is visible. */
  press: { type: "spring", bounce: 0, duration: 0.24 },
  /** Controls, indicators, selection. */
  snap: { type: "spring", bounce: 0, duration: 0.3 },
  /** Reposition and layout change. Apple ships damping 1.0 / response 0.4 for
   *  PiP-style movement; this is that. */
  move: { type: "spring", bounce: 0, duration: 0.4 },
  /** Drawers and sheets. Apple ships damping 0.8 / response 0.3. */
  sheet: { type: "spring", bounce: 0.2, duration: 0.3 },
  /** Post-flick release only — never for something that arrived on its own. */
  fling: { type: "spring", bounce: 0.2, duration: 0.4 },
} as const satisfies Record<string, Spring>;

export type SpringName = keyof typeof SPRING;

/**
 * Reduced-motion equivalents.
 *
 * Not "no animation" — a gentler, non-vestibular one. State still changes
 * visibly (opacity, colour); what goes away is travel and overshoot, which are
 * the parts that cause discomfort. Every preset collapses to the same short
 * linear cross-fade so nothing reads as slower or faster than anything else.
 */
const REDUCED: Record<SpringName, { duration: number; ease: "linear" }> = {
  press: { duration: 0.08, ease: "linear" },
  snap: { duration: 0.12, ease: "linear" },
  move: { duration: 0.12, ease: "linear" },
  sheet: { duration: 0.12, ease: "linear" },
  fling: { duration: 0.12, ease: "linear" },
};

/**
 * Every component reads its springs through this hook rather than importing
 * `SPRING` directly, so reduced-motion support is structural — a component
 * cannot forget to handle it, because there is no path that skips it.
 *
 * `travel` is the companion: it returns the transform values a component wants
 * to animate, or their no-op equivalents under reduced motion, so callers do
 * not each hand-roll a conditional.
 */
export function useSprings() {
  const reduced = useReducedMotion();

  return useMemo(() => {
    const spring = (name: SpringName) =>
      reduced ? REDUCED[name] : SPRING[name];

    /** Collapse a travel distance to 0 when motion is reduced. */
    const travel = (px: number) => (reduced ? 0 : px);

    /** Collapse a scale factor to 1 when motion is reduced. */
    const scale = (factor: number) => (reduced ? 1 : factor);

    return { reduced: Boolean(reduced), spring, travel, scale };
  }, [reduced]);
}

/**
 * Apple's momentum projection, from the *Designing Fluid Interfaces* sample.
 *
 * A flick should land where the gesture was *going*, not at whatever boundary
 * happened to be nearest when the finger left the glass. This is the same
 * exponential-decay model scroll deceleration uses.
 *
 * Note this is NOT the physics-textbook `v² / 2a` — that is a different curve
 * and does not match the platform's feel.
 *
 * @param velocity px/s at release
 * @param decelerationRate 0.998 for normal scroll feel, 0.99 for snappier
 * @returns the additional distance the gesture would travel on its own
 */
export function project(velocity: number, decelerationRate = 0.998): number {
  return ((velocity / 1000) * decelerationRate) / (1 - decelerationRate);
}

/**
 * Progressive boundary resistance.
 *
 * A hard stop at an edge reads as frozen — as though the interface stopped
 * listening. Resistance that grows the further you push reads as responsive,
 * but with nothing more to show. Real things slow before they stop.
 *
 * @param overshoot how far past the boundary the pointer has travelled
 * @param dimension the size of the dragged surface, which sets how much give it has
 */
export function rubberband(
  overshoot: number,
  dimension: number,
  constant = 0.55,
): number {
  return (
    (overshoot * dimension * constant) /
    (dimension + constant * Math.abs(overshoot))
  );
}

type Sample = { value: number; time: number };

/** Samples older than this are stale — a flick is a recent event, and averaging
 *  in older points flattens exactly the peak we are trying to measure. */
const VELOCITY_WINDOW_MS = 100;
const MAX_SAMPLES = 6;

/**
 * Tracks a short position/time history so release velocity is measured over a
 * window rather than differenced between the last two events.
 *
 * A two-point difference is dominated by whatever the final pointermove
 * happened to be — often a near-zero delta as the finger settles, which reads
 * as "the user stopped" when they actually threw it.
 */
export function useVelocityTracker() {
  const samples = useRef<Sample[]>([]);

  const reset = useCallback((value = 0, time = performance.now()) => {
    samples.current = [{ value, time }];
  }, []);

  const record = useCallback((value: number, time = performance.now()) => {
    const next = samples.current;
    next.push({ value, time });
    while (
      next.length > MAX_SAMPLES ||
      (next.length > 2 && time - next[0].time > VELOCITY_WINDOW_MS)
    ) {
      next.shift();
    }
  }, []);

  /** @returns px/s, signed. 0 when there is not enough history to be honest. */
  const velocity = useCallback((): number => {
    const s = samples.current;
    if (s.length < 2) return 0;
    const last = s[s.length - 1];
    const first = s[0];
    const dt = last.time - first.time;
    if (dt <= 0) return 0;
    return ((last.value - first.value) / dt) * 1000;
  }, []);

  return { reset, record, velocity };
}

/**
 * Where a flick will come to rest, given where it is and how fast it is going.
 * Pair with a spring whose initial `velocity` is the same release velocity, so
 * there is no visible seam between dragging and animating.
 */
export function projectedEndpoint(
  current: number,
  releaseVelocity: number,
  decelerationRate = 0.998,
): number {
  return current + project(releaseVelocity, decelerationRate);
}

/** Nearest value in `points` to `target` — used to pick a snap position from a
 *  projected endpoint rather than from the raw release position. */
export function nearestSnapPoint(target: number, points: number[]): number {
  return points.reduce((best, p) =>
    Math.abs(p - target) < Math.abs(best - target) ? p : best,
  );
}

/** Movement below this is a tap, not a drag. Also the distance a press can
 *  wander before it is treated as cancelled (§10). */
export const HYSTERESIS_PX = 10;
