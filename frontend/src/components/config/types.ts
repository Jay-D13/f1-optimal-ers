import type React from 'react';

export interface ConfigSectionCardProps {
    icon: React.ReactNode;
    title: string;
    status?: string;
    summary?: string;
    accent?: 'neutral' | 'accent' | 'ok' | 'warn';
    className?: string;
    headerRight?: React.ReactNode;
    children: React.ReactNode;
}

export interface FieldMeta {
    id: string;
    label: string;
    hint?: string;
    error?: string;
}

export interface SegmentedOption<T extends string> {
    value: T;
    label: string;
    description?: string;
}

export interface ValidationState {
    isValid: boolean;
    errors: string[];
    warnings: string[];
}
