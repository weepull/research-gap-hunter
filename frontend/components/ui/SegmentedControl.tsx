"use client";

import { motion } from "motion/react";
import { useRef } from "react";
import { useSprings } from "@/lib/motion";

/**
 * A choice among a few named options.
 *
 * Replaces two native `<select>`s and, on the gaps page, an `<input
 * type="range">` over a three-item array. The range was the worse of the two:
 * three discrete named options are not a continuous quantity, and because the
 * input's value was the array *index*, screen readers announced "0, 1, 2"
 * rather than "5, 10, 20". A radiogroup states the actual choices.
 *
 * Keyboard model is the standard one for radiogroups — a single tab stop on
 * the selected option, arrows to move (wrapping), Home/End for the ends.
 * Selection follows focus, which is correct here because every option is
 * cheap to preview and there is no destructive commit.
 */
export default function SegmentedControl<T extends string | number>({
  options,
  value,
  onChange,
  label,
  size = "md",
}: {
  options: ReadonlyArray<{ value: T; label: string }>;
  value: T;
  onChange: (value: T) => void;
  label: string;
  size?: "sm" | "md";
}) {
  const { spring } = useSprings();
  const refs = useRef<Array<HTMLButtonElement | null>>([]);
  const index = options.findIndex((o) => o.value === value);

  function move(to: number) {
    const next = (to + options.length) % options.length;
    onChange(options[next].value);
    refs.current[next]?.focus();
  }

  function onKeyDown(e: React.KeyboardEvent) {
    switch (e.key) {
      case "ArrowLeft":
      case "ArrowUp":
        e.preventDefault();
        move(index - 1);
        break;
      case "ArrowRight":
      case "ArrowDown":
        e.preventDefault();
        move(index + 1);
        break;
      case "Home":
        e.preventDefault();
        move(0);
        break;
      case "End":
        e.preventDefault();
        move(options.length - 1);
        break;
    }
  }

  const pad = size === "sm" ? "px-2.5 py-1" : "px-3.5 py-1.5";
  const type = size === "sm" ? "text-caption" : "text-subhead";

  return (
    <div
      role="radiogroup"
      aria-label={label}
      onKeyDown={onKeyDown}
      className="inline-flex items-center gap-0.5 rounded-sm border border-hairline bg-surface-inset p-0.5"
    >
      {options.map((option, i) => {
        const selected = option.value === value;
        return (
          <button
            key={String(option.value)}
            ref={(el) => {
              refs.current[i] = el;
            }}
            type="button"
            role="radio"
            aria-checked={selected}
            // One tab stop for the whole group, per the radiogroup pattern.
            tabIndex={selected ? 0 : -1}
            onPointerDown={() => onChange(option.value)}
            onClick={() => onChange(option.value)}
            className={`relative rounded-xs ${pad} ${type} transition-colors ${
              selected ? "text-label" : "text-label-3 hover:text-label-2"
            }`}
          >
            {selected && (
              // One indicator that slides between options, rather than a
              // background appearing under each — the movement is what shows
              // the relationship between the old choice and the new one.
              <motion.span
                layoutId={`segmented-${label}`}
                transition={spring("snap")}
                className="absolute inset-0 rounded-xs bg-surface shadow-[var(--shadow-1)] ring-1 ring-hairline"
                aria-hidden="true"
              />
            )}
            <span className="relative">{option.label}</span>
          </button>
        );
      })}
    </div>
  );
}
