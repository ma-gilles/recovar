import { AnyAtom } from '@tanstack/store';
export declare function useStore<TAtom extends AnyAtom | undefined, T>(atom: TAtom, selector: (snapshot: TAtom extends {
    get: () => infer TSnapshot;
} ? TSnapshot : undefined) => T, compare?: (a: T, b: T) => boolean): T;
