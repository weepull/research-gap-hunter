"use client";

import Pressable from "@/components/ui/Pressable";

/**
 * The search input.
 *
 * The 500ms debounce behind this field paces the backend; it is not input
 * latency, and the field itself responds to every keystroke immediately. The
 * busy state is raised on the first character rather than when the request
 * finally goes out, so the wait is never silent.
 */
export default function SearchField({
  value,
  onChange,
  busy,
  placeholder = "Describe a problem…",
}: {
  value: string;
  onChange: (value: string) => void;
  busy?: boolean;
  placeholder?: string;
}) {
  return (
    <div className="relative flex-1">
      <input
        type="search"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        autoFocus
        aria-label="Search limitation statements"
        // Placeholder uses the 4.5:1 label colour, not the decorative one:
        // placeholder text is text, and it carries the only instruction about
        // what to type here.
        //
        // The UA search-cancel button is suppressed in favour of the explicit
        // clear button below, which is reachable by keyboard and has a label.
        className="search-field w-full rounded-sm border border-hairline bg-surface px-4 py-2.5 pr-24 text-body text-label shadow-[var(--shadow-1)] placeholder:text-label-3"
      />
      <div className="absolute inset-y-0 right-2 flex items-center gap-1">
        {busy && (
          <span className="text-caption text-label-3" aria-hidden="true">
            Searching…
          </span>
        )}
        {value && (
          <Pressable
            onClick={() => onChange("")}
            aria-label="Clear search"
            pressScale={0.9}
            className="flex h-6 w-6 items-center justify-center rounded-full bg-surface-inset text-label-2"
          >
            <span aria-hidden="true">×</span>
          </Pressable>
        )}
      </div>
    </div>
  );
}
