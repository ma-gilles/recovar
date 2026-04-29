const require_constants = require("../constants.cjs");
let seroval = require("seroval");
//#region src/ssr/serializer/transformer.ts
/**
* Create a strongly-typed serialization adapter for SSR hydration.
* Use to register custom types with the router serializer.
*/
function createSerializationAdapter(opts) {
	return opts;
}
/** Create a Seroval plugin for server-side serialization only. */
function makeSsrSerovalPlugin(serializationAdapter, options) {
	return (0, seroval.createPlugin)({
		tag: "$TSR/t/" + serializationAdapter.key,
		test: serializationAdapter.test,
		parse: { stream(value, ctx) {
			return ctx.parse(serializationAdapter.toSerializable(value));
		} },
		serialize(node, ctx) {
			options.didRun = true;
			return require_constants.GLOBAL_TSR + ".t.get(\"" + serializationAdapter.key + "\")(" + ctx.serialize(node) + ")";
		},
		deserialize: void 0
	});
}
/** Create a Seroval plugin for client/server symmetric (de)serialization. */
function makeSerovalPlugin(serializationAdapter) {
	return (0, seroval.createPlugin)({
		tag: "$TSR/t/" + serializationAdapter.key,
		test: serializationAdapter.test,
		parse: {
			sync(value, ctx) {
				return ctx.parse(serializationAdapter.toSerializable(value));
			},
			async async(value, ctx) {
				return await ctx.parse(serializationAdapter.toSerializable(value));
			},
			stream(value, ctx) {
				return ctx.parse(serializationAdapter.toSerializable(value));
			}
		},
		serialize: void 0,
		deserialize(node, ctx) {
			return serializationAdapter.fromSerializable(ctx.deserialize(node));
		}
	});
}
//#endregion
exports.createSerializationAdapter = createSerializationAdapter;
exports.makeSerovalPlugin = makeSerovalPlugin;
exports.makeSsrSerovalPlugin = makeSsrSerovalPlugin;

//# sourceMappingURL=transformer.cjs.map