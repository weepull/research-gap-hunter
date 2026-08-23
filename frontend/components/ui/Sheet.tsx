"use client";

import { AnimatePresence, animate, motion, useMotionValue } from "motion/react";
import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
} from "react";
import { createPortal } from "react-dom";
import {
  HYSTERESIS_PX,
  projectedEndpoint,
  rubberband,
  useSprings,
  useVelocityTracker,
} from "@/lib/motion";

/**
 * A draggable detail sheet.
 *
 * This is where most of the interaction principles in this codebase actually
 * get exercised, so the behaviour is worth stating explicitly:
 *
 *  - **1:1 tracking that respects the grab offset.** The sheet stays glued to
 *    the finger from wherever it was grabbed. Snapping to a fixed point on
 *    grab breaks the illusion immediately.
 *  - **Pointer capture**, so tracking survives the pointer leaving the sheet.
 *  - **Rubber-banding upward.** Dragging past the top resists progressively
 *    instead of stopping dead. A hard stop reads as frozen; resistance reads as
 *    "still listening, but there's nothing more here".
 *  - **Momentum projection on release.** Where the sheet *lands* is decided
 *    from the projected endpoint of the flick, not from the release position,
 *    so a fast short flick dismisses and a slow long drag may not.
 *  - **Velocity handoff.** The settle animation starts at the finger's exact
 *    release velocity, so there is no seam between dragging and animating.
 *  - **Interruptible.** Grabbing a sheet mid-dismiss stops the animation at its
 *    current on-screen value and hands control straight back to the finger. It
 *    does not finish closing and reopen.
 *
 * Accessibility is not layered on afterwards: focus is trapped while open,
 * Escape closes, focus returns to whatever opened it, the background is made
 * `inert` so it cannot be reached by tab or screen reader, and under reduced
 * motion the whole thing cross-fades in place instead of travelling.
 */

/** Fraction of sheet height the projected endpoint must pass to dismiss. */
const DISMISS_FRACTION = 0.35;
/** Below this speed a flick is not a flick, and position alone decides. */
const FLICK_VELOCITY = 520;

export default function Sheet({
  open,
  onClose,
  title,
  children,
  labelledBy,
}: {
  open: boolean;
  onClose: () => void;
  title: string;
  children: ReactNode;
  labelledBy?: string;
}) {
  const { spring, reduced } = useSprings();
  const y = useMotionValue(0);
  const tracker = useVelocityTracker();
  const sheetRef = useRef<HTMLDivElement | null>(null);
  const heightRef = useRef(0);
  const grabOffset = useRef(0);
  const dragging = useRef(false);
  const restoreFocus = useRef<HTMLElement | null>(null);
  const [mounted, setMounted] = useState(false);

  useEffect(() => setMounted(true), []);

  // Drag progress drives the scrim and the background's push-back through a
  // CSS custom property. Doing it this way keeps both continuous with the
  // gesture — they respond while the finger moves, not only when it lifts —
  // without threading a motion value across the portal boundary.
  const publishProgress = useCallback(
    (value: number) => {
      const h = heightRef.current || 1;
      const p = Math.max(0, Math.min(1, 1 - value / h));
      document.documentElement.style.setProperty("--sheet-progress", p.toFixed(3));
    },
    [],
  );

  useEffect(() => {
    const unsubscribe = y.on("change", publishProgress);
    return () => unsubscribe();
  }, [y, publishProgress]);

  // Background inert + scroll lock while open.
  useEffect(() => {
    if (!open) return;
    restoreFocus.current = document.activeElement as HTMLElement | null;

    const siblings = Array.from(document.body.children).filter(
      (el) => !el.hasAttribute("data-sheet-root"),
    );
    siblings.forEach((el) => el.setAttribute("inert", ""));

    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    document.documentElement.setAttribute("data-sheet-open", "");

    return () => {
      siblings.forEach((el) => el.removeAttribute("inert"));
      document.body.style.overflow = prevOverflow;
      document.documentElement.removeAttribute("data-sheet-open");
      document.documentElement.style.removeProperty("--sheet-progress");
      restoreFocus.current?.focus?.();
    };
  }, [open]);

  // Escape closes; Tab is trapped within the sheet.
  useEffect(() => {
    if (!open) return;
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
        return;
      }
      if (e.key !== "Tab") return;
      const root = sheetRef.current;
      if (!root) return;
      const focusable = root.querySelectorAll<HTMLElement>(
        'a[href], button:not([disabled]), input, select, textarea, [tabindex]:not([tabindex="-1"])',
      );
      if (focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    }
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [open, onClose]);

  // Move focus into the sheet once it exists.
  useEffect(() => {
    if (open) sheetRef.current?.focus();
  }, [open]);

  function onPointerDown(e: ReactPointerEvent<HTMLDivElement>) {
    if (reduced) return; // No drag-to-dismiss when motion is reduced.
    const el = sheetRef.current;
    if (!el) return;
    heightRef.current = el.getBoundingClientRect().height;

    // Interruption: stop whatever is animating and continue from the value
    // that is actually on screen right now, never from the target.
    y.stop();

    dragging.current = true;
    grabOffset.current = e.clientY - y.get();
    el.setPointerCapture(e.pointerId);
    tracker.reset(e.clientY);
  }

  function onPointerMove(e: ReactPointerEvent<HTMLDivElement>) {
    if (!dragging.current) return;
    tracker.record(e.clientY);
    const raw = e.clientY - grabOffset.current;
    // Downward is free; upward resists more the further it is pushed.
    y.set(raw >= 0 ? raw : -rubberband(-raw, heightRef.current || 1));
  }

  function endDrag(e: ReactPointerEvent<HTMLDivElement>) {
    if (!dragging.current) return;
    dragging.current = false;
    sheetRef.current?.releasePointerCapture?.(e.pointerId);

    const velocity = tracker.velocity();
    const current = y.get();
    const height = heightRef.current || 1;

    // Land where the gesture was going, not where the finger happened to stop.
    const endpoint = projectedEndpoint(current, velocity);
    const flicked = Math.abs(velocity) > FLICK_VELOCITY;
    const dismiss = flicked
      ? velocity > 0
      : endpoint > height * DISMISS_FRACTION;

    if (dismiss) {
      // Velocity handoff: the animation continues at the finger's speed.
      animate(y, height, { ...spring("fling"), velocity }).then(() => onClose());
    } else {
      animate(y, 0, { ...spring("fling"), velocity });
    }
  }

  if (!mounted) return null;

  return createPortal(
    <AnimatePresence>
      {open && (
        <div data-sheet-root className="fixed inset-0 z-[100]">
          {/* Scrim. Dims to focus; its opacity tracks the drag so the
              relationship between gesture and background is continuous. */}
          <motion.button
            type="button"
            aria-label="Close detail"
            onClick={onClose}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={spring("snap")}
            className="sheet-scrim absolute inset-0 w-full cursor-default"
          />

          <motion.div
            ref={sheetRef}
            role="dialog"
            aria-modal="true"
            aria-label={labelledBy ? undefined : title}
            aria-labelledby={labelledBy}
            tabIndex={-1}
            style={{ y }}
            // Enters and exits along the same path: up from the bottom, back
            // down to the bottom. A surface that arrives one way and leaves
            // another reads as two unrelated things.
            initial={reduced ? { opacity: 0 } : { y: "100%" }}
            animate={reduced ? { opacity: 1 } : { y: 0 }}
            exit={reduced ? { opacity: 0 } : { y: "100%" }}
            transition={spring("sheet")}
            className="material-thick absolute inset-x-0 bottom-0 mx-auto flex max-h-[86vh] w-full max-w-3xl flex-col rounded-t-xl focus:outline-none"
          >
            {/* The handle is the grab target, and it looks like one. */}
            <div
              onPointerDown={onPointerDown}
              onPointerMove={onPointerMove}
              onPointerUp={endDrag}
              onPointerCancel={endDrag}
              className="shrink-0 cursor-grab touch-none px-5 pt-3 pb-1 active:cursor-grabbing"
            >
              <div
                aria-hidden="true"
                className="mx-auto h-1 w-10 rounded-full bg-hairline-strong"
              />
            </div>

            <div className="min-h-0 flex-1 overflow-y-auto px-5 pb-7 pt-2 sm:px-6">
              {children}
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>,
    document.body,
  );
}

/** Movement below this is a tap on the handle, not a drag. Exported so callers
 *  reasoning about the same threshold do not invent a second number. */
export { HYSTERESIS_PX };
