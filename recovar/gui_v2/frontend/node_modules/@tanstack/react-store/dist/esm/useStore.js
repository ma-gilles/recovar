import { useCallback } from "react";
import { useSyncExternalStoreWithSelector } from "use-sync-external-store/shim/with-selector";
function defaultCompare(a, b) {
  return a === b;
}
function useStore(atom, selector, compare = defaultCompare) {
  const subscribe = useCallback(
    (handleStoreChange) => {
      if (!atom) {
        return () => {
        };
      }
      const { unsubscribe } = atom.subscribe(handleStoreChange);
      return unsubscribe;
    },
    [atom]
  );
  const boundGetSnapshot = useCallback(() => atom?.get(), [atom]);
  const selectedSnapshot = useSyncExternalStoreWithSelector(
    subscribe,
    boundGetSnapshot,
    boundGetSnapshot,
    selector,
    compare
  );
  return selectedSnapshot;
}
export {
  useStore
};
//# sourceMappingURL=useStore.js.map
