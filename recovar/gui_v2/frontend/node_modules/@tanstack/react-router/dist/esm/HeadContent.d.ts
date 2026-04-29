import { AssetCrossOriginConfig } from '@tanstack/router-core';
export interface HeadContentProps {
    assetCrossOrigin?: AssetCrossOriginConfig;
}
/**
 * Render route-managed head tags (title, meta, links, styles, head scripts).
 * Place inside the document head of your app shell.
 * @link https://tanstack.com/router/latest/docs/framework/react/guide/document-head-management
 */
export declare function HeadContent(props: HeadContentProps): import("react/jsx-runtime").JSX.Element;
