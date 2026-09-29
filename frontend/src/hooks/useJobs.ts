import { useCallback, useEffect, useRef, useState } from 'react';
import { api, type Job, type RunSettings } from '../api';

const POLL_MS = 1500;

/**
 * The server's run jobs, polled while one is running. `version` goes up each time a run finishes, so pages can
 * reload their data.
 */
export function useJobs() {
    const [jobs, setJobs] = useState<Job[]>([]);
    const [version, setVersion] = useState(0);
    const [error, setError] = useState<string | null>(null);
    const statuses = useRef<Record<string, Job['status']>>({});

    const refresh = useCallback(async () => {
        try {
            const next = await api.jobs();
            let finished = false;
            for (const job of next) {
                if (statuses.current[job.id] === 'running' && job.status !== 'running') finished = true;
                statuses.current[job.id] = job.status;
            }
            setJobs(next);
            setError(null);
            if (finished) setVersion((v) => v + 1);
        } catch (e) {
            setError((e as Error).message);
        }
    }, []);

    const running = jobs.find((j) => j.status === 'running') ?? null;
    const runningId = running?.id ?? null;

    useEffect(() => {
        const timer = window.setTimeout(refresh, 0);
        return () => window.clearTimeout(timer);
    }, [refresh]);

    useEffect(() => {
        if (!runningId) return;
        const timer = window.setInterval(refresh, POLL_MS);
        return () => window.clearInterval(timer);
    }, [runningId, refresh]);

    const start = useCallback(async (round: number, settings: RunSettings) => {
        const job = await api.startJob(round, settings);
        statuses.current[job.id] = job.status;
        setJobs((prev) => [job, ...prev]);
        return job;
    }, []);

    const cancel = useCallback(async (id: string) => {
        await api.cancelJob(id);
        await refresh();
    }, [refresh]);

    return { jobs, running, version, error, start, cancel };
}
