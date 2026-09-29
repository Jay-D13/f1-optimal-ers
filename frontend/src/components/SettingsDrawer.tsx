import { useEffect } from 'react';
import { X } from 'lucide-react';
import { DEFAULT_SETTINGS, type Round, type RunSettings } from '../api';
import { Legend } from './Chrome';

interface SettingsDrawerProps {
    open: boolean;
    onClose: () => void;
    round: Round;
    settings: RunSettings;
    onChange: (next: RunSettings) => void;
}

/** MGU-K deploy limit (kW) against speed (km/h), FIA C5.2.8. */
function overtakeCurve(v: number) {
    return v < 355 ? Math.min(350, 7100 - 20 * v) : 0;
}
function standardCurve(v: number) {
    if (v < 340) return Math.min(350, 1800 - 5 * v);
    if (v < 345) return 6900 - 20 * v;
    return 0;
}

function DeployCurves({ session }: { session: RunSettings['session'] }) {
    const w = 400, h = 150, x0 = 36, x1 = w - 6, y0 = 10, y1 = h - 22;
    const sx = (v: number) => x0 + (v / 360) * (x1 - x0);
    const sy = (p: number) => y1 - (p / 350) * (y1 - y0);
    const path = (f: (v: number) => number) => {
        let d = '';
        for (let v = 0; v <= 360; v += 2) d += `${v ? 'L' : 'M'}${sx(v).toFixed(1)},${sy(f(v)).toFixed(1)}`;
        return d;
    };
    const quali = session === 'qualifying';
    return (
        <div className="flex flex-col gap-2">
            <svg width="100%" viewBox={`0 0 ${w} ${h}`} role="img" aria-label="MGU-K deploy limit against speed">
                <line x1={x0} x2={x1} y1={y1} y2={y1} stroke="var(--rule)" />
                <line x1={x0} x2={x1} y1={y0} y2={y0} stroke="var(--rule)" strokeDasharray="3 4" />
                <text x={x0 - 6} y={y0 + 4} textAnchor="end" className="tick">350</text>
                <text x={x0 - 6} y={y1 + 4} textAnchor="end" className="tick">0</text>
                {[0, 100, 200, 300].map((v) => (
                    <text key={v} x={sx(v)} y={h - 6} textAnchor="middle" className="tick">{v}</text>
                ))}
                <text x={x1} y={h - 6} textAnchor="end" className="tick">km/h</text>
                <path d={path(standardCurve)} fill="none" stroke={quali ? 'var(--mute)' : 'var(--accent)'} strokeWidth={quali ? 1.5 : 2.5} strokeDasharray={quali ? '5 4' : undefined} />
                <path d={path(overtakeCurve)} fill="none" stroke={quali ? 'var(--accent)' : 'var(--mute)'} strokeWidth={quali ? 2.5 : 1.5} strokeDasharray={quali ? undefined : '5 4'} />
            </svg>
            <Legend items={quali
                ? [{ label: 'Overtake (qualifying, always on)', color: 'var(--accent)' }, { label: 'Standard (race)', color: 'var(--mute)', dashed: true }]
                : [{ label: 'Standard (race)', color: 'var(--accent)' }, { label: 'Overtake (when triggered)', color: 'var(--mute)', dashed: true }]} />
        </div>
    );
}

function Rules({ round, settings }: { round: Round; settings: RunSettings }) {
    if (settings.regulations === '2025') {
        return (
            <div>
                <div className="row-rule"><span>MGU-K power</span><span className="num">120 kW</span></div>
                <div className="row-rule"><span>Recovery per lap</span><span className="num">2 MJ</span></div>
                <div className="row-rule"><span>Deployment per lap</span><span className="num">4 MJ</span></div>
                <p className="mt-3 text-sm text-mute">2014–2025 rules on this event's track, for comparison.</p>
            </div>
        );
    }
    const quali = settings.session === 'qualifying';
    return (
        <div className="flex flex-col gap-5">
            <div>
                <div className="row-rule">
                    <span>Recharge cap per lap</span>
                    <span className="num font-bold">{(quali ? round.quali_cap_mj : round.race_cap_mj).toFixed(1)} MJ</span>
                </div>
                <div className="row-rule">
                    <span>{quali ? 'Race' : 'Qualifying'} cap, for reference</span>
                    <span className="num text-mute">{(quali ? round.race_cap_mj : round.quali_cap_mj).toFixed(1)} MJ</span>
                </div>
                <div className="row-rule"><span>Charge window (max − min)</span><span className="num">4 MJ</span></div>
                <div className="row-rule"><span>Harvest at full throttle (super-clip)</span><span className="num">{round.superclip_kw} kW</span></div>
                <div className="row-rule"><span>Ramp-down after the first step</span><span className="num">{round.ramp_kw_s} kW/s</span></div>
                <div className="row-rule"><span>Power-limited distance</span><span className="num">{round.power_limited_m} m</span></div>
                {!round.verified && (
                    <p className="mt-2 text-sm text-mute">This event's cap isn't in a public FIA document; the value is a best estimate.</p>
                )}
                {quali && (
                    <p className="mt-2 text-sm text-mute">
                        The out-lap refills the battery, so the lap starts full. The run-up from the last apex to the line isn't timed.
                    </p>
                )}
            </div>
            <div className="flex flex-col gap-2">
                <span className="cap" style={{ color: 'var(--ink)' }}>MGU-K deploy limit · C5.2.8</span>
                <DeployCurves session={settings.session} />
            </div>
            <div className="flex flex-col gap-2">
                <span className="cap" style={{ color: 'var(--ink)' }}>Straight Mode zones</span>
                {round.straight_mode_zones.length > 0 ? (
                    <div className="flex flex-wrap gap-1.5">
                        {round.straight_mode_zones.map((z) => (
                            <span key={`${z.corner}-${z.offset_m}`} className="num border-2 border-ink px-2 py-0.5 text-xs">
                                T{z.corner} +{z.offset_m} m
                            </span>
                        ))}
                    </div>
                ) : (
                    <span className="text-sm text-mute">None at this event.</span>
                )}
                <p className="text-sm text-mute">Each zone starts this far after the corner and ends at the next one. The model decides where to open the wings inside it.</p>
            </div>
        </div>
    );
}

export function SettingsDrawer({ open, onClose, round, settings, onChange }: SettingsDrawerProps) {
    useEffect(() => {
        if (!open) return;
        const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
        window.addEventListener('keydown', onKey);
        return () => window.removeEventListener('keydown', onKey);
    }, [open, onClose]);

    if (!open) return null;
    const set = <K extends keyof RunSettings>(key: K, value: RunSettings[K]) => onChange({ ...settings, [key]: value });
    const race = settings.session === 'race';

    return (
        <div className="fixed inset-0 z-40 flex justify-end" role="dialog" aria-modal="true" aria-label="Run settings">
            <button type="button" className="absolute inset-0 cursor-default" style={{ background: 'rgba(0,0,0,0.35)' }} aria-label="Close settings" onClick={onClose} />
            <aside className="relative flex h-full w-full max-w-[480px] flex-col overflow-y-auto border-l-2 border-ink bg-paper">
                <div className="sticky top-0 z-10 flex items-center justify-between border-b-2 border-ink bg-paper px-6 py-4">
                    <div>
                        <div className="cap">Settings · R{String(round.round).padStart(2, '0')}</div>
                        <div className="display text-3xl">{round.name}</div>
                    </div>
                    <button type="button" className="btn" style={{ padding: 8 }} onClick={onClose} aria-label="Close settings"><X size={16} /></button>
                </div>

                <section className="flex flex-col gap-4 border-b-2 border-ink px-6 py-5">
                    <h2 className="display text-2xl">The rules</h2>
                    <Rules round={round} settings={settings} />
                </section>

                <section className="flex flex-col gap-4 px-6 py-5">
                    <div className="flex items-center justify-between">
                        <h2 className="display text-2xl">The run</h2>
                        <button type="button" className="link text-sm" onClick={() => onChange({ ...DEFAULT_SETTINGS, session: settings.session, regulations: settings.regulations })}>
                            Reset to defaults
                        </button>
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                        {race ? (
                            <>
                                <div className="field">
                                    <label htmlFor="laps">Laps</label>
                                    <input id="laps" type="number" min={1} max={20} value={settings.laps} onChange={(e) => set('laps', Math.max(1, Number(e.target.value) || 1))} />
                                </div>
                                <div className="field">
                                    <label htmlFor="soc0">Start charge</label>
                                    <input id="soc0" type="number" min={0} max={1} step={0.05} value={settings.initial_soc} onChange={(e) => set('initial_soc', Number(e.target.value))} />
                                    <span className="hint">0–1 of the battery</span>
                                </div>
                                <div className="field">
                                    <label htmlFor="soc1">Min end charge</label>
                                    <input id="soc1" type="number" min={0} max={1} step={0.05} value={settings.final_soc_min} onChange={(e) => set('final_soc_min', Number(e.target.value))} />
                                </div>
                            </>
                        ) : (
                            <div className="col-span-2 text-sm text-mute">Qualifying: one flying lap. Start and end charge follow the rules above.</div>
                        )}
                        <div className="field">
                            <label htmlFor="ds">Step (m)</label>
                            <input id="ds" type="number" min={1} max={50} step={0.5} value={settings.ds} onChange={(e) => set('ds', Number(e.target.value) || DEFAULT_SETTINGS.ds)} />
                            <span className="hint">5 m reads about 0.2 % fast; smaller is slower to solve</span>
                        </div>
                        <div className="field">
                            <label htmlFor="colloc">Integration</label>
                            <select id="colloc" value={settings.collocation} onChange={(e) => set('collocation', e.target.value as RunSettings['collocation'])}>
                                <option value="trapezoidal">Trapezoidal</option>
                                <option value="hermite_simpson">Hermite–Simpson</option>
                                <option value="euler">Euler (rough)</option>
                            </select>
                        </div>
                        <div className="field">
                            <label htmlFor="hess">Hessian</label>
                            <select id="hess" value={settings.ipopt_hessian} onChange={(e) => set('ipopt_hessian', e.target.value as RunSettings['ipopt_hessian'])}>
                                <option value="exact">Exact</option>
                                <option value="limited-memory">Limited memory</option>
                            </select>
                            {settings.ipopt_hessian === 'limited-memory' && <span className="hint">Faster, but can stop far from the optimum</span>}
                        </div>
                        {race && (
                            <div className="field">
                                <label htmlFor="tyre">Tyre model</label>
                                <select id="tyre" value={settings.tire_model} onChange={(e) => set('tire_model', e.target.value as RunSettings['tire_model'])}>
                                    <option value="scalar">Constant grip</option>
                                    <option value="dynamic">Temperature and wear</option>
                                </select>
                                {settings.tire_model === 'dynamic' && settings.laps < 2 && <span className="hint">Needs two laps or more</span>}
                            </div>
                        )}
                        {race && settings.tire_model === 'dynamic' && (
                            <div className="field">
                                <label htmlFor="compound">Compound</label>
                                <select id="compound" value={settings.tire_compound} onChange={(e) => set('tire_compound', e.target.value as RunSettings['tire_compound'])}>
                                    <option value="soft">Soft</option>
                                    <option value="medium">Medium</option>
                                    <option value="hard">Hard</option>
                                </select>
                            </div>
                        )}
                    </div>
                </section>
            </aside>
        </div>
    );
}
