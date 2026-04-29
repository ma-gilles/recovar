const require_runtime = require("./_virtual/_rolldown/runtime.cjs");
let react = require("react");
react = require_runtime.__toESM(react);
//#region src/matchContext.tsx
var matchContext = react.createContext(void 0);
var dummyMatchContext = react.createContext(void 0);
//#endregion
exports.dummyMatchContext = dummyMatchContext;
exports.matchContext = matchContext;

//# sourceMappingURL=matchContext.cjs.map