Object.defineProperty(exports, Symbol.toStringTag, { value: "Module" });
//#region src/isServer/server.ts
var isServer = process.env.NODE_ENV === "test" ? void 0 : true;
//#endregion
exports.isServer = isServer;

//# sourceMappingURL=server.cjs.map