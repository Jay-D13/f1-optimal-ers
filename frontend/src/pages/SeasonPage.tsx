import { useEffect, useMemo, useState } from 'react';
import { Play } from 'lucide-react';
import { api, type Job, type Round, type RunDetail } from '../api';
import { ApiError, Legend, Stripes, ThemeToggle } from '../components/Chrome';
import { Trace } from '../components/Trace';
import { driverCode, formatDelta, formatLap, formatTimestamp, median } from '../format';

interface SeasonPageProps {
    rounds: Round[] | null;
    error: string | null;
    running: Job | null;
    onOpen: (round: number) => void;
    onRun: (round: number) => void;
}

function CapBlocks({ mj }: { mj: number }) {
    // One block per 1 MJ, up to the 9 MJ largest cap; a half block for .5
    return (
        <span className="flex items-center gap-[3px]" aria-hidden="true">
            {Array.from({ length: 9 }, (_, i) => {
                const fill = Math.min(Math.max(mj - i, 0), 1);
                return (
                    <span key={i} className="relative h-3 w-[7px]" style={{ background: 'var(--rule)' }}>
                        {fill > 0 && <span className="absolute inset-y-0 left-0" style={{ width: `${fill * 100}%`, background: 'var(--ink)' }} />}
                    </span>
                );
            })}
        </span>
    );
}

function LatestRunCard({ round }: { round: Round | null }) {
    const [detail, setDetail] = useState<RunDetail | null>(null);
    const run = round?.latest_run ?? null;

    useEffect(() => {
        if (!round || !run) return;
        let live = true;
        api.run(round.track, run.run_id, round.round).then((d) => live && setDetail(d)).catch(() => live && setDetail(null));
        return () => { live = false; };
    }, [round, run]);

    if (!round || !run) {
        return (
            <div className="box flex flex-col gap-2 p-4">
                <span className="cap">Latest run</span>
                <p className="text-sm">No 2026 qualifying runs yet. Pick a round with a racing line and press Run.</p>
            </div>
        );
    }
    const delta = run.delta_to_pole;
    const length = detail?.summary.track_info?.total_length ?? detail?.series.s.at(-1) ?? 1;
    const lines = detail ? [
        ...(detail.pole_trace ? [{ s: detail.pole_trace.s, y: detail.pole_trace.v.map((v) => v * 3.6), color: 'var(--pole)' }] : []),
        { s: detail.series.s, y: detail.series.v.map((v) => v * 3.6), color: 'var(--model)' },
    ] : [];

    return (
        <div className="box flex flex-col gap-3 p-4">
            <span className="cap">Latest run · R{String(round.round).padStart(2, '0')} {round.circuit} · {formatTimestamp(run.timestamp)}</span>
            <h2 className="display text-4xl">
                {delta == null ? formatLap(run.lap_time) : `${Math.abs(delta).toFixed(2)} s ${delta < 0 ? 'under' : 'over'} ${round.pole?.driver}`}
            </h2>
            {detail && (
                <Trace title="Speed" unit="km/h" lines={lines} length={length} yMin={50} yMax={360} ticks={[100, 200, 300]}
                    height={140} cursor={null} onCursor={() => undefined} />
            )}
            <Legend items={[{ label: 'model', color: 'var(--model)' }, { label: 'pole lap', color: 'var(--pole)' }]} />
        </div>
    );
}

export function SeasonPage({ rounds, error, running, onOpen, onRun }: SeasonPageProps) {
    const stats = useMemo(() => {
        const list = rounds ?? [];
        const gaps = list
            .map((r) => (r.latest_run?.delta_to_pole != null && r.pole ? Math.abs(r.latest_run.delta_to_pole) / r.pole.time * 100 : null))
            .filter((g): g is number => g != null);
        const latest = list
            .filter((r) => r.latest_run)
            .sort((a, b) => (b.latest_run!.timestamp ?? '').localeCompare(a.latest_run!.timestamp ?? ''))[0] ?? null;
        return {
            medianGap: median(gaps),
            runCount: gaps.length,
            runnable: list.filter((r) => r.has_raceline).length,
            total: list.length,
            missing: list.filter((r) => !r.has_raceline).map((r) => r.circuit),
            latest,
        };
    }, [rounds]);

    return (
        <div className="min-h-screen">
            <Stripes />
            <div className="mx-auto flex max-w-[1440px] flex-col gap-7 px-6 pb-10 pt-10 lg:px-12">
                <header className="flex flex-wrap items-end justify-between gap-8">
                    <div className="flex flex-col gap-3">
                        <div className="flex items-center gap-4">
                            <span className="cap">ERS Pole Lab · 2026 qualifying · {stats.total} rounds</span>
                            <ThemeToggle />
                        </div>
                        <h1 className="display text-6xl lg:text-7xl">Can the model<br />find pole?</h1>
                    </div>
                    <div className="flex flex-wrap items-end gap-10">
                        <div className="flex flex-col gap-1.5">
                            <span className="cap">median gap to pole</span>
                            <span className="display text-6xl text-accent">{stats.medianGap == null ? '–' : `${stats.medianGap.toFixed(1)}%`}</span>
                            <span className="num text-xs text-mute">{stats.runCount} of {stats.runnable} rounds run</span>
                        </div>
                        <div className="flex flex-col gap-1.5">
                            <span className="cap">runnable rounds</span>
                            <span className="display text-6xl">{stats.runnable}/{stats.total}</span>
                            <span className="num text-xs text-mute">{stats.total - stats.runnable} need a racing line</span>
                        </div>
                    </div>
                </header>

                {error && <ApiError message={error} />}

                <div className="flex flex-col gap-7 xl:flex-row">
                    <section className="box min-w-0 flex-1 overflow-x-auto" aria-label="2026 rounds">
                        <table className="w-full min-w-[860px] border-collapse text-[15px]">
                            <thead>
                                <tr className="border-b-2 border-ink text-left">
                                    {['Rd', 'Event', 'Quali cap · MJ', 'Real pole', 'Model', 'Δ s', ''].map((h) => (
                                        <th key={h} scope="col" className="cap h-9 px-3 font-normal first:pl-4">{h}</th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                {(rounds ?? []).map((r) => {
                                    const isRunning = running?.round === r.round;
                                    const run = r.latest_run;
                                    return (
                                        <tr key={r.round} className="h-11 border-b border-rule last:border-b-0" style={isRunning ? { background: 'var(--sel)' } : undefined}>
                                            <td className="num pl-4 pr-3 text-mute">R{String(r.round).padStart(2, '0')}</td>
                                            <td className="px-3">
                                                <button type="button" className="link text-left font-semibold" onClick={() => onOpen(r.round)}>{r.name}</button>
                                                <span className="ml-2.5 text-[13px] text-mute">{r.circuit}</span>
                                            </td>
                                            <td className="px-3">
                                                <span className="flex items-center gap-2.5"><CapBlocks mj={r.quali_cap_mj} /><span className="num">{r.quali_cap_mj.toFixed(1)}{r.verified ? '' : '*'}</span></span>
                                            </td>
                                            <td className="px-3">
                                                {r.pole ? <><span className="num">{formatLap(r.pole.time)}</span><span className="num ml-2.5 text-xs text-mute">{driverCode(r.pole.driver)}</span></> : '–'}
                                            </td>
                                            <td className="px-3">
                                                {isRunning ? <span className="num pulse">running…</span>
                                                    : run ? <span className="num font-bold text-model">{formatLap(run.lap_time)}</span>
                                                    : r.has_raceline ? <span className="num text-mute">not run</span>
                                                    : <span className="text-xs text-mute">no racing line</span>}
                                            </td>
                                            <td className="num px-3 font-bold">{run && !isRunning ? formatDelta(run.delta_to_pole) : <span className="font-normal text-mute">–</span>}</td>
                                            <td className="px-3 pr-4 text-right">
                                                {run && !isRunning ? (
                                                    <button type="button" className="btn btn-go" onClick={() => onOpen(r.round)}>Open</button>
                                                ) : (
                                                    <button type="button" className="btn" disabled={!r.has_raceline || running != null} onClick={() => onRun(r.round)}
                                                        title={!r.has_raceline ? 'No racing line for the current layout yet' : running ? 'A run is in progress' : undefined}>
                                                        <Play size={12} fill="currentColor" /> Run
                                                    </button>
                                                )}
                                            </td>
                                        </tr>
                                    );
                                })}
                            </tbody>
                        </table>
                    </section>

                    <aside className="flex w-full shrink-0 flex-col gap-5 xl:w-[440px]">
                        <LatestRunCard round={stats.latest} />
                        <div className="box flex flex-col gap-2.5 p-4">
                            <span className="cap">Notes</span>
                            {stats.missing.length > 0 && (
                                <p className="text-sm leading-relaxed">No racing line yet for {stats.missing.join(', ')}.</p>
                            )}
                            {(rounds ?? []).some((r) => !r.verified) && (
                                <p className="text-sm leading-relaxed">* Cap not from a public FIA document.</p>
                            )}
                            <p className="text-sm leading-relaxed">Poles from Jolpica, cross-checked with F1 timing. The model is one generic 2026 car on each event's own energy rules.</p>
                        </div>
                    </aside>
                </div>
            </div>
        </div>
    );
}
