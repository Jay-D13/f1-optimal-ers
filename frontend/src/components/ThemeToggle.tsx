import { Moon, Sun } from 'lucide-react';
import { useTheme } from '../context/ThemeContext';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
    return twMerge(clsx(inputs));
}

export function ThemeToggle({ className }: { className?: string }) {
    const { theme, toggleTheme } = useTheme();

    return (
        <button
            onClick={toggleTheme}
            className={cn(
                "p-2 rounded-full transition-colors hover:bg-black/10 dark:hover:bg-white/10",
                "text-retro-text",
                className
            )}
            aria-label="Toggle Dark Mode"
        >
            {theme === 'light' ? <Moon size={20} /> : <Sun size={20} />}
        </button>
    );
}
