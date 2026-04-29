export type AssetCrossOrigin = 'anonymous' | 'use-credentials';
export type AssetCrossOriginConfig = AssetCrossOrigin | Partial<Record<'modulepreload' | 'stylesheet', AssetCrossOrigin>>;
export type ManifestAssetLink = string | {
    href: string;
    crossOrigin?: AssetCrossOrigin;
};
export declare function getAssetCrossOrigin(assetCrossOrigin: AssetCrossOriginConfig | undefined, kind: 'modulepreload' | 'stylesheet'): AssetCrossOrigin | undefined;
export declare function resolveManifestAssetLink(link: ManifestAssetLink): {
    href: string;
    crossOrigin?: AssetCrossOrigin;
};
export type Manifest = {
    routes: Record<string, {
        filePath?: string;
        preloads?: Array<ManifestAssetLink>;
        assets?: Array<RouterManagedTag>;
    }>;
};
export type RouterManagedTag = {
    tag: 'title';
    attrs?: Record<string, any>;
    children: string;
} | {
    tag: 'meta' | 'link';
    attrs?: Record<string, any>;
    children?: never;
} | {
    tag: 'script';
    attrs?: Record<string, any>;
    children?: string;
} | {
    tag: 'style';
    attrs?: Record<string, any>;
    children?: string;
};
