"use strict";
Object.defineProperty(exports, Symbol.toStringTag, { value: "Module" });
const react = require("react");
const withSelector = require("use-sync-external-store/shim/with-selector");
function defaultCompare(a, b) {
  return a === b;
}
function useStore(atom, selector, compare = defaultCompare) {
  const subscribe = react.useCallback(
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
  const boundGetSnapshot = react.useCallback(() => atom?.get(), [atom]);
  const selectedSnapshot = withSelector.useSyncExternalStoreWithSelector(
    subscribe,
    boundGetSnapshot,
    boundGetSnapshot,
    selector,
    compare
  );
  return selectedSnapshot;
}
exports.useStore = useStore;
//# sourceMappingURL=useStore.cjs.map
