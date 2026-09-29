import { Moon, Sun } from 'lucide-react';
import { useTheme } from '../context/ThemeContext';

/** The programme's three coloured stripes across the top of the page. */
export function Stripes() {
    return (
        <div className="stripes" aria-hidden="true">
            <span style={{ background: 'var(--stripe-1)' }} />
            <span style={{ background: 'var(--stripe-2)' }} />
            <span style={{ background: 'var(--stripe-3)' }} />
        </div>
    );
}

export function ThemeToggle() {
    const { theme, toggleTheme } = useTheme();
    const dark = theme === 'dark';
    return (
        <button type="button" className="btn" onClick={toggleTheme} aria-label={dark ? 'Light mode' : 'Dark mode'} title={dark ? 'Light mode' : 'Dark mode'} style={{ padding: 8 }}>
            {dark ? <Sun size={16} /> : <Moon size={16} />}
        </button>
    );
}

export function Legend({ items }: { items: { label: string; color: string; dashed?: boolean }[] }) {
    return (
        <div className="num flex flex-wrap gap-4 text-xs text-mute">
            {items.map((item) => (
                <span key={item.label} className="flex items-center gap-1.5">
                    <span
                        style={item.dashed
                            ? { width: 16, borderTop: `2px dashed ${item.color}` }
                            : { width: 16, height: 4, background: item.color }}
                    />
                    {item.label}
                </span>
            ))}
        </div>
    );
}

export function ApiError({ message }: { message: string }) {
    return (
        <div className="box p-4" role="alert">
            <div className="display text-2xl">No connection to the API</div>
            <p className="mt-2 text-sm">
                {message}. Start it with <code className="num">./start.sh</code> from the repository root.
            </p>
        </div>
    );
}
