import { Readable } from '@tanstack/react-store';
import { GetStoreConfig } from '@tanstack/router-core';
declare module '@tanstack/router-core' {
    interface RouterReadableStore<TValue> extends Readable<TValue> {
    }
}
export declare const getStoreFactory: GetStoreConfig;
