import type { FieldMeta } from './types';

interface FieldHintProps {
    meta: Pick<FieldMeta, 'id' | 'hint' | 'error'>;
}

function FieldHint({ meta }: FieldHintProps) {
    if (!meta.hint && !meta.error) {
        return null;
    }

    return (
        <div id={`${meta.id}-hint`} className="mt-1 min-h-[1rem] font-mono text-[10px] uppercase tracking-wide">
            {meta.error ? <span className="text-amber-600 dark:text-amber-400">{meta.error}</span> : <span className="text-retro-text/60">{meta.hint}</span>}
        </div>
    );
}

export default FieldHint;
