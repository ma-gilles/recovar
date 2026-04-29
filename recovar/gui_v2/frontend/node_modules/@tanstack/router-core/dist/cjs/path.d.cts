import { LRUCache } from './lru-cache.cjs';
/** Join path segments, cleaning duplicate slashes between parts. */
export declare function joinPaths(paths: Array<string | undefined>): string;
/** Remove repeated slashes from a path string. */
export declare function cleanPath(path: string): string;
/** Trim leading slashes (except preserving root '/'). */
export declare function trimPathLeft(path: string): string;
/** Trim trailing slashes (except preserving root '/'). */
export declare function trimPathRight(path: string): string;
/** Trim both leading and trailing slashes. */
export declare function trimPath(path: string): string;
/** Remove a trailing slash from value when appropriate for comparisons. */
export declare function removeTrailingSlash(value: string, basepath: string): string;
/**
 * Compare two pathnames for exact equality after normalizing trailing slashes
 * relative to the provided `basepath`.
 */
export declare function exactPathTest(pathName1: string, pathName2: string, basepath: string): boolean;
interface ResolvePathOptions {
    base: string;
    to: string;
    trailingSlash?: 'always' | 'never' | 'preserve';
    cache?: LRUCache<string, string>;
}
/**
 * Resolve a destination path against a base, honoring trailing-slash policy
 * and supporting relative segments (`.`/`..`) and absolute `to` values.
 */
export declare function resolvePath({ base, to, trailingSlash, cache, }: ResolvePathOptions): string;
/**
 * Create a pre-compiled decode config from allowed characters.
 * This should be called once at router initialization.
 */
export declare function compileDecodeCharMap(pathParamsAllowedCharacters: ReadonlyArray<string>): (encoded: string) => string;
interface InterpolatePathOptions {
    path?: string;
    params: Record<string, unknown>;
    /**
     * A function that decodes a path parameter value.
     * Obtained from `compileDecodeCharMap(pathParamsAllowedCharacters)`.
     */
    decoder?: (encoded: string) => string;
}
type InterPolatePathResult = {
    interpolatedPath: string;
    usedParams: Record<string, unknown>;
    isMissingParams: boolean;
};
/**
 * Interpolate params and wildcards into a route path template.
 *
 * - Encodes params safely (configurable allowed characters)
 * - Supports `{-$optional}` segments, `{prefix{$id}suffix}` and `{$}` wildcards
 */
export declare function interpolatePath({ path, params, decoder, ...rest }: InterpolatePathOptions): InterPolatePathResult;
export {};
