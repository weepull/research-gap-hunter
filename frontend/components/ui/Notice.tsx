import type { ReactNode } from "react";

/**
 * Information that is not a failure.
 *
 * Used for states the system reached on purpose — a feature switched off for
 * this deployment, a result that is legitimately empty. These read as calm
 * statements rather than as errors, because presenting a deliberate state in
 * red trains people to distrust output that is working correctly.
 */
export default function Notice({
  children,
  tone = "neutral",
  title,
}: {
  children: ReactNode;
  tone?: "neutral" | "caution" | "negative";
  title?: string;
}) {
  const tones = {
    neutral: "border-hairline bg-surface-sunken text-label-2",
    caution: "border-caution-line bg-caution-wash text-caution",
    negative: "border-negative-line bg-negative-wash text-negative",
  } as const;

  return (
    <div className={`rounded-md border px-4 py-3 text-callout ${tones[tone]}`}>
      {title && <p className="mb-1 text-headline">{title}</p>}
      {children}
    </div>
  );
}
