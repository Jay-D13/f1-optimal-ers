import { clsx } from 'clsx';
import type { SegmentedOption } from './types';

interface SegmentedControlProps<T extends string> {
    value: T;
    options: SegmentedOption<T>[];
    onChange: (next: T) => void;
    className?: string;
}

function SegmentedControl<T extends string>({ value, options, onChange, className }: SegmentedControlProps<T>) {
    return (
        <div className={clsx('grid grid-cols-1 gap-2 sm:grid-cols-2', className)} role="radiogroup" aria-label="segmented-control">
            {options.map((option) => {
                const active = option.value === value;
                return (
                    <button
                        key={option.value}
                        type="button"
                        role="radio"
                        aria-checked={active}
                        onClick={() => onChange(option.value)}
                        className={clsx(
                            'group rounded-lg border px-3 py-2 text-left font-mono text-sm transition-all duration-150',
                            active
                                ? 'border-f1-red bg-f1-black text-f1-white shadow-[0_8px_16px_-12px_rgba(0,0,0,0.75)] dark:bg-white dark:text-black'
                                : 'border-panel-border bg-panel-muted/70 text-retro-text hover:border-f1-red/45 hover:bg-panel-muted',
                        )}
                    >
                        <div className="font-bold tracking-tight">{option.label}</div>
                        {option.description && (
                            <div className={clsx('mt-1 text-[11px] leading-tight', active ? 'text-f1-white/75 dark:text-black/70' : 'text-retro-text/65')}>
                                {option.description}
                            </div>
                        )}
                    </button>
                );
            })}
        </div>
    );
}

export default SegmentedControl;
