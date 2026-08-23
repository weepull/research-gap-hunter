"use client";

import Pressable from "@/components/ui/Pressable";

/**
 * A genuine failure — something went wrong that the user did not choose.
 *
 * Kept visually distinct from EmptyState and Notice on purpose. If a correct
 * empty result and a dead backend look the same, neither can be trusted.
 */
export default function ErrorState({
  title,
  message,
  onRetry,
}: {
  title: string;
  message: string;
  onRetry?: () => void;
}) {
  return (
    <div
      role="alert"
      className="rounded-lg border border-negative-line bg-negative-wash px-6 py-8 text-center"
    >
      <h3 className="text-title-3 text-negative">{title}</h3>
      <p className="mx-auto mt-2 max-w-[56ch] text-callout text-label-2">{message}</p>
      {onRetry && (
        <div className="mt-5 flex justify-center">
          <Pressable
            onClick={onRetry}
            className="rounded-sm border border-hairline-strong bg-surface px-4 py-2 text-headline text-label shadow-[var(--shadow-1)]"
          >
            Try again
          </Pressable>
        </div>
      )}
    </div>
  );
}
