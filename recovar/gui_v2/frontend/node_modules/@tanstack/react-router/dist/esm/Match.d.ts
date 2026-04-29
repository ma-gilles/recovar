import * as React from 'react';
export declare const Match: React.MemoExoticComponent<({ matchId, }: {
    matchId: string;
}) => import("react/jsx-runtime").JSX.Element>;
export declare const MatchInner: React.MemoExoticComponent<({ matchId, }: {
    matchId: string;
}) => any>;
/**
 * Render the next child match in the route tree. Typically used inside
 * a route component to render nested routes.
 *
 * @link https://tanstack.com/router/latest/docs/framework/react/api/router/outletComponent
 */
export declare const Outlet: React.MemoExoticComponent<() => import("react/jsx-runtime").JSX.Element | null>;
