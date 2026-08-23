import type { ReactNode } from "react";
import Eyebrow from "@/components/ui/Eyebrow";

/**
 * A designed answer, not a shrug.
 *
 * Every empty state in this app has a *reason*, and the reason is the content:
 * cross-domain matching returning nothing means no pair cleared the measured
 * noise floor, which is correct output. Saying so is the difference between an
 * interface that looks broken and one that looks certain.
 */
export default function EmptyState({
  eyebrow,
  title,
  children,
  action,
}: {
  eyebrow?: string;
  title: string;
  children?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="rounded-lg border border-dashed border-hairline-strong bg-surface-sunken px-6 py-10 text-center">
      {eyebrow && <Eyebrow className="mb-2">{eyebrow}</Eyebrow>}
      <h3 className="text-title-3 text-label">{title}</h3>
      {children && (
        <div className="mx-auto mt-2 max-w-[52ch] text-callout text-label-2">
          {children}
        </div>
      )}
      {action && <div className="mt-5 flex justify-center">{action}</div>}
    </div>
  );
}
