import { clsx } from 'clsx';
import type { ConfigSectionCardProps } from './types';

const accentClasses: Record<NonNullable<ConfigSectionCardProps['accent']>, string> = {
    neutral: 'border-panel-border',
    accent: 'border-f1-red/35 shadow-[0_6px_24px_-18px_rgba(225,6,0,0.65)]',
    ok: 'border-emerald-500/35 shadow-[0_6px_24px_-18px_rgba(16,185,129,0.65)]',
    warn: 'border-amber-500/35 shadow-[0_6px_24px_-18px_rgba(245,158,11,0.65)]',
};

function ConfigSectionCard({
    icon,
    title,
    status,
    summary,
    accent = 'neutral',
    className,
    headerRight,
    children,
}: ConfigSectionCardProps) {
    return (
        <section
            className={clsx(
                'config-card rounded-xl border bg-panel-bg/90 p-4 md:p-5 backdrop-blur-sm transition-all duration-200',
                accentClasses[accent],
                className,
            )}
        >
            <div className="mb-4 flex flex-wrap items-center justify-between gap-3 border-b border-panel-border/70 pb-3">
                <div className="flex items-center gap-2 text-retro-text">
                    <span className="text-f1-red">{icon}</span>
                    <h3 className="font-mono text-sm font-bold uppercase tracking-wide">{title}</h3>
                </div>
                {(summary || status || headerRight) && (
                    <div className="flex items-center gap-2">
                        {summary && (
                            <span className="rounded-full border border-panel-border bg-panel-muted px-2 py-0.5 font-mono text-[10px] uppercase tracking-wide text-retro-text/70">
                                {summary}
                            </span>
                        )}
                        {status && (
                            <span className="rounded-full border border-panel-border/80 bg-retro-text/5 px-2 py-0.5 font-mono text-[10px] uppercase tracking-wide text-retro-text/70">
                                {status}
                            </span>
                        )}
                        {headerRight}
                    </div>
                )}
            </div>
            {children}
        </section>
    );
}

export default ConfigSectionCard;
