import { useCallback, useEffect, useState } from 'react';
import { api, DEFAULT_SETTINGS, type Round, type RunSettings } from './api';
import { useJobs } from './hooks/useJobs';
import { LapPage } from './pages/LapPage';
import { SeasonPage } from './pages/SeasonPage';

const SETTINGS_KEY = 'run-settings';

/** The round in the URL hash (#/round/13), or null for the season page. */
function roundFromHash(): number | null {
    const match = window.location.hash.match(/^#\/round\/(\d+)$/);
    return match ? Number(match[1]) : null;
}

function loadSettings(): RunSettings {
    try {
        const saved = localStorage.getItem(SETTINGS_KEY);
        if (saved) return { ...DEFAULT_SETTINGS, ...JSON.parse(saved) };
    } catch {
        // Storage unavailable or corrupt: defaults
    }
    return DEFAULT_SETTINGS;
}

function App() {
    const [roundNumber, setRoundNumber] = useState<number | null>(roundFromHash);
    const [rounds, setRounds] = useState<Round[] | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [settings, setSettings] = useState<RunSettings>(loadSettings);
    const { jobs, running, version, start, cancel } = useJobs();

    useEffect(() => {
        const onHash = () => setRoundNumber(roundFromHash());
        window.addEventListener('hashchange', onHash);
        return () => window.removeEventListener('hashchange', onHash);
    }, []);

    useEffect(() => {
        api.season().then((r) => { setRounds(r); setError(null); }).catch((e: Error) => setError(e.message));
    }, [version]);

    useEffect(() => {
        try {
            localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
        } catch {
            // Not saved: fine
        }
    }, [settings]);

    const open = useCallback((round: number | null) => {
        window.location.hash = round == null ? '' : `/round/${round}`;
        window.scrollTo(0, 0);
    }, []);

    const runFromSeason = useCallback(async (round: number) => {
        try {
            await start(round, { ...settings, session: 'qualifying', regulations: '2026', laps: 1 });
        } catch (e) {
            setError((e as Error).message);
        }
    }, [start, settings]);

    const round = rounds?.find((r) => r.round === roundNumber) ?? null;

    if (roundNumber != null && round) {
        return (
            <LapPage key={round.round} round={round} settings={settings} onSettingsChange={setSettings}
                jobs={jobs} running={running} version={version} onStart={start} onCancel={cancel} onBack={() => open(null)} />
        );
    }
    return <SeasonPage rounds={rounds} error={error} running={running} onOpen={open} onRun={runFromSeason} />;
}

export default App;
