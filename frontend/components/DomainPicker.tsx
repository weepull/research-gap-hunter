"use client";

import SegmentedControl from "@/components/ui/SegmentedControl";
import { DOMAINS } from "@/lib/api";

/** The domain choice, in the two-option form the corpus actually supports. */
export default function DomainPicker({
  value,
  onChange,
  label = "Domain",
}: {
  value: string;
  onChange: (value: string) => void;
  label?: string;
}) {
  return (
    <SegmentedControl
      label={label}
      value={value}
      onChange={onChange}
      options={DOMAINS.map((d) => ({ value: d.value, label: d.label }))}
    />
  );
}
