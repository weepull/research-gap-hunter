"use client";

import Link from "next/link";
import { motion } from "motion/react";
import {
  forwardRef,
  useCallback,
  useRef,
  useState,
  type ComponentPropsWithoutRef,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
} from "react";
import { HYSTERESIS_PX, useSprings } from "@/lib/motion";

const MotionLink = motion.create(Link);

/**
 * The press-feedback primitive. Everything interactive in the app goes through
 * it, so "responds instantly" is a property of the system rather than something
 * each component has to remember.
 *
 * Two rules it exists to enforce:
 *
 * 1. **Feedback on pointer-down, not on release.** The moment feedback waits
 *    for `click`, directness falls off a cliff — the element reads as dead for
 *    the whole duration of the press. Here the scale change is applied in the
 *    `pointerdown` handler, before anything else happens.
 *
 * 2. **Cancel by dragging away, and restore on return.** A press that slides
 *    off the target releases, and sliding back re-arms it. Without this a
 *    slightly shaky tap looks like the interface ignored it.
 *
 * Commit stays on the native `click` event rather than on `pointerup`. That
 * keeps keyboard activation, right-click, cmd-click and every assistive-tech
 * path working exactly as the platform intends — the pointer handlers here
 * drive *appearance* only, never behaviour.
 */

type Feedback = "scale" | "opacity" | "none";

type BaseProps = {
  children: ReactNode;
  className?: string;
  /** How the press reads. `opacity` suits text links, where scaling type is
   *  distracting; `none` suits elements that show their own pressed styling
   *  via the `data-pressed` attribute. */
  feedback?: Feedback;
  /** Press depth. 0.97 by default — enough to register, not enough to wobble
   *  the layout. Large surfaces use a shallower value because the same ratio
   *  moves far more pixels at the edges. */
  pressScale?: number;
  disabled?: boolean;
};

type ButtonProps = BaseProps &
  Omit<ComponentPropsWithoutRef<"button">, keyof BaseProps> & {
    as?: "button";
    href?: never;
  };

type LinkProps = BaseProps &
  Omit<ComponentPropsWithoutRef<"a">, keyof BaseProps> & {
    as: "link";
    href: string;
  };

type AnchorProps = BaseProps &
  Omit<ComponentPropsWithoutRef<"a">, keyof BaseProps> & {
    as: "a";
    href: string;
  };

type DivProps = BaseProps &
  Omit<ComponentPropsWithoutRef<"div">, keyof BaseProps> & {
    as: "div";
    href?: never;
  };

export type PressableProps = ButtonProps | LinkProps | AnchorProps | DivProps;

function usePressState(disabled: boolean) {
  const [pressed, setPressed] = useState(false);
  const origin = useRef<{ x: number; y: number } | null>(null);

  const onPointerDown = useCallback(
    (e: ReactPointerEvent<Element>) => {
      if (disabled) return;
      // Secondary buttons open context menus; they are not a press.
      if (e.button !== 0 && e.pointerType === "mouse") return;
      origin.current = { x: e.clientX, y: e.clientY };
      // Capture so tracking survives the pointer leaving the element's bounds —
      // otherwise the drag-away/return behaviour below can never fire.
      (e.currentTarget as Element).setPointerCapture?.(e.pointerId);
      setPressed(true);
    },
    [disabled],
  );

  const onPointerMove = useCallback((e: ReactPointerEvent<Element>) => {
    const start = origin.current;
    if (!start) return;
    const dx = e.clientX - start.x;
    const dy = e.clientY - start.y;
    const within = Math.hypot(dx, dy) <= HYSTERESIS_PX * 2;
    // Re-arms on return, rather than latching off at the first wobble.
    setPressed(within);
  }, []);

  const release = useCallback(() => {
    origin.current = null;
    setPressed(false);
  }, []);

  return { pressed, onPointerDown, onPointerMove, release };
}

export const Pressable = forwardRef<HTMLElement, PressableProps>(
  function Pressable(props, ref) {
    const {
      as = "button",
      children,
      className = "",
      feedback = "scale",
      pressScale = 0.97,
      disabled = false,
      ...rest
    } = props as BaseProps & {
      as?: "button" | "link" | "a" | "div";
      [key: string]: unknown;
    };

    const { spring, reduced } = useSprings();
    const { pressed, onPointerDown, onPointerMove, release } =
      usePressState(disabled);

    // Under reduced motion the scale is dropped but the opacity dip is kept:
    // reduced motion means non-vestibular feedback, not the absence of it.
    const animate =
      feedback === "none"
        ? {}
        : feedback === "opacity" || reduced
          ? { opacity: pressed ? 0.62 : 1, scale: 1 }
          : { scale: pressed ? pressScale : 1, opacity: 1 };

    // Built without `ref` so the object is never a carrier for it — the ref
    // is applied as a JSX attribute on each branch instead.
    const shared = {
      className,
      animate,
      transition: spring("press"),
      "data-pressed": pressed || undefined,
      onPointerDown,
      onPointerMove,
      onPointerUp: release,
      onPointerCancel: release,
      onLostPointerCapture: release,
      // `will-change` only while a press is actually in flight. A standing
      // hint promotes a layer for every interactive element on the page and
      // costs more than it saves.
      style: pressed ? { willChange: "transform, opacity" } : undefined,
      ...rest,
    };

    if (as === "link") {
      const p = shared as unknown as ComponentPropsWithoutRef<typeof MotionLink>;
      return (
        <MotionLink ref={ref as React.Ref<HTMLAnchorElement>} {...p}>
          {children}
        </MotionLink>
      );
    }

    if (as === "a") {
      const p = shared as unknown as ComponentPropsWithoutRef<typeof motion.a>;
      return (
        <motion.a ref={ref as React.Ref<HTMLAnchorElement>} {...p}>
          {children}
        </motion.a>
      );
    }

    if (as === "div") {
      const p = shared as unknown as ComponentPropsWithoutRef<typeof motion.div>;
      return (
        <motion.div ref={ref as React.Ref<HTMLDivElement>} {...p}>
          {children}
        </motion.div>
      );
    }

    const p = shared as unknown as ComponentPropsWithoutRef<typeof motion.button>;
    return (
      <motion.button
        ref={ref as React.Ref<HTMLButtonElement>}
        type="button"
        disabled={disabled}
        {...p}
      >
        {children}
      </motion.button>
    );
  },
) as (
  props: PressableProps & { ref?: React.Ref<HTMLElement> },
) => React.ReactElement;

export default Pressable;
