Object.defineProperty(exports, Symbol.toStringTag, { value: "Module" });
const require_RouterServer = require("./RouterServer.cjs");
const require_renderRouterToString = require("./renderRouterToString.cjs");
const require_defaultRenderHandler = require("./defaultRenderHandler.cjs");
const require_renderRouterToStream = require("./renderRouterToStream.cjs");
const require_defaultStreamHandler = require("./defaultStreamHandler.cjs");
exports.RouterServer = require_RouterServer.RouterServer;
exports.defaultRenderHandler = require_defaultRenderHandler.defaultRenderHandler;
exports.defaultStreamHandler = require_defaultStreamHandler.defaultStreamHandler;
exports.renderRouterToStream = require_renderRouterToStream.renderRouterToStream;
exports.renderRouterToString = require_renderRouterToString.renderRouterToString;
var _tanstack_router_core_ssr_server = require("@tanstack/router-core/ssr/server");
Object.keys(_tanstack_router_core_ssr_server).forEach(function(k) {
	if (k !== "default" && !Object.prototype.hasOwnProperty.call(exports, k)) Object.defineProperty(exports, k, {
		enumerable: true,
		get: function() {
			return _tanstack_router_core_ssr_server[k];
		}
	});
});
