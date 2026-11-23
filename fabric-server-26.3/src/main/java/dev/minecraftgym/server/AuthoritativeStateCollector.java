package dev.minecraftgym.server;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentLinkedQueue;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;

import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.effect.MobEffectInstance;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.level.Level;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.phys.Vec3;

final class AuthoritativeStateCollector {
    static final int CAPACITY = 2400;

    private final Pseudonymizer pseudonymizer;
    private final ArrayDeque<JsonObject> records = new ArrayDeque<>(CAPACITY);
    private final ConcurrentLinkedQueue<JsonObject> pendingEvents = new ConcurrentLinkedQueue<>();
    private final ConcurrentLinkedQueue<BlockProbe> pendingBlockProbes = new ConcurrentLinkedQueue<>();
    private final Map<String, Map<String, Integer>> priorInventory = new HashMap<>();
    private final Map<String, Float> priorHealth = new HashMap<>();
    private long nextSequence;

    AuthoritativeStateCollector(Pseudonymizer pseudonymizer) {
        this.pseudonymizer = pseudonymizer;
    }

    void onJoin(ServerPlayer player) {
        pendingEvents.add(event("player_join", pseudonymizer.actorId(player.getUUID())));
    }

    void onDisconnect(ServerPlayer player) {
        String actorId = pseudonymizer.actorId(player.getUUID());
        pendingEvents.add(event("player_disconnect", actorId));
        priorInventory.remove(actorId);
        priorHealth.remove(actorId);
    }

    void onBlockBreak(Player player, Level level, BlockPos position, BlockState state) {
        if (level.isClientSide()) {
            return;
        }
        JsonObject event = playerEvent("block_broken", player);
        addBlockContext(event, level, position);
        event.addProperty("block_before", BuiltInRegistries.BLOCK.getKey(state.getBlock()).toString());
        event.addProperty(
                "block_after",
                BuiltInRegistries.BLOCK.getKey(level.getBlockState(position).getBlock()).toString());
        pendingEvents.add(event);
    }

    void onBlockUse(Player player, Level level, BlockPos position, Direction direction) {
        onBlockInteraction("block_used", player, level, position);
        if (level.isClientSide()) {
            return;
        }
        String actorId = pseudonymizer.actorId(player.getUUID());
        probeBlock(actorId, level, position);
        probeBlock(actorId, level, position.relative(direction));
    }

    void onBlockInteraction(String type, Player player, Level level, BlockPos position) {
        if (level.isClientSide()) {
            return;
        }
        JsonObject event = playerEvent(type, player);
        addBlockContext(event, level, position);
        event.addProperty(
                "block", BuiltInRegistries.BLOCK.getKey(level.getBlockState(position).getBlock()).toString());
        pendingEvents.add(event);
    }

    synchronized void capture(MinecraftServer server) {
        JsonObject record = new JsonObject();
        record.addProperty("interface", "minecraft-authoritative-tick/v1");
        record.addProperty("sequence", nextSequence++);
        record.addProperty("server_tick", server.getTickCount());
        record.addProperty("monotonic_ns", System.nanoTime());

        JsonArray actors = new JsonArray();
        JsonArray events = new JsonArray();
        JsonObject pending;
        while ((pending = pendingEvents.poll()) != null) {
            events.add(pending);
        }
        appendBlockChanges(events);
        for (ServerPlayer player : server.getPlayerList().getPlayers()) {
            String actorId = pseudonymizer.actorId(player.getUUID());
            actors.add(actor(player, actorId));
            detectEvents(player, actorId, events);
        }
        record.add("actors", actors);
        record.add("events", events);
        records.addLast(record);
        while (records.size() > CAPACITY) {
            records.removeFirst();
        }
    }

    synchronized Batch after(long sequence, int limit) {
        int boundedLimit = Math.max(1, Math.min(limit, 400));
        long earliest = records.isEmpty() ? nextSequence : records.getFirst().get("sequence").getAsLong();
        long latest = records.isEmpty() ? nextSequence - 1 : records.getLast().get("sequence").getAsLong();
        boolean gap = sequence < earliest - 1;
        List<JsonObject> selected = new ArrayList<>();
        for (JsonObject record : records) {
            if (record.get("sequence").getAsLong() > sequence) {
                selected.add(record.deepCopy());
                if (selected.size() >= boundedLimit) {
                    break;
                }
            }
        }
        return new Batch(earliest, latest, gap, selected);
    }

    private JsonObject actor(ServerPlayer player, String actorId) {
        JsonObject value = new JsonObject();
        value.addProperty("actor_id", actorId);
        value.addProperty("dimension", player.level().dimension().identifier().toString());
        value.addProperty("x", player.getX());
        value.addProperty("y", player.getY());
        value.addProperty("z", player.getZ());
        value.addProperty("yaw", player.getYRot());
        value.addProperty("pitch", player.getXRot());
        Vec3 velocity = player.getDeltaMovement();
        value.addProperty("velocity_x", velocity.x);
        value.addProperty("velocity_y", velocity.y);
        value.addProperty("velocity_z", velocity.z);
        value.addProperty("health", player.getHealth());
        value.addProperty("max_health", player.getMaxHealth());
        value.addProperty("hunger", player.getFoodData().getFoodLevel());
        value.addProperty("armor", player.getArmorValue());
        value.addProperty("air", player.getAirSupply());
        value.addProperty("on_ground", player.onGround());
        value.addProperty("on_fire", player.isOnFire());
        value.addProperty("underwater", player.isUnderWater());
        value.add("inventory", inventoryArray(player));
        value.add("effects", effects(player));
        return value;
    }

    private void detectEvents(ServerPlayer player, String actorId, JsonArray events) {
        float health = player.getHealth();
        Float previousHealth = priorHealth.put(actorId, health);
        if (previousHealth != null && previousHealth > 0 && health <= 0) {
            events.add(event("death", actorId));
        }

        Map<String, Integer> current = inventoryTotals(player);
        Map<String, Integer> previous = priorInventory.put(actorId, current);
        if (previous == null) {
            return;
        }
        Set<String> items = new HashSet<>(previous.keySet());
        items.addAll(current.keySet());
        for (String item : items) {
            int before = previous.getOrDefault(item, 0);
            int after = current.getOrDefault(item, 0);
            if (before != after) {
                JsonObject event = event("inventory_delta", actorId);
                event.addProperty("item", item);
                event.addProperty("before", before);
                event.addProperty("after", after);
                event.addProperty("delta", after - before);
                events.add(event);
            }
        }
    }

    private static JsonArray inventoryArray(ServerPlayer player) {
        JsonArray inventory = new JsonArray();
        for (int slot = 0; slot < player.getInventory().getContainerSize(); slot++) {
            ItemStack stack = player.getInventory().getItem(slot);
            JsonObject item = new JsonObject();
            item.addProperty("slot", slot);
            item.addProperty("item", stack.isEmpty()
                    ? "minecraft:air"
                    : BuiltInRegistries.ITEM.getKey(stack.getItem()).toString());
            item.addProperty("count", stack.isEmpty() ? 0 : stack.getCount());
            inventory.add(item);
        }
        return inventory;
    }

    private static Map<String, Integer> inventoryTotals(ServerPlayer player) {
        Map<String, Integer> totals = new HashMap<>();
        for (int slot = 0; slot < player.getInventory().getContainerSize(); slot++) {
            ItemStack stack = player.getInventory().getItem(slot);
            if (!stack.isEmpty()) {
                totals.merge(
                        BuiltInRegistries.ITEM.getKey(stack.getItem()).toString(),
                        stack.getCount(),
                        Integer::sum);
            }
        }
        return Map.copyOf(totals);
    }

    private static JsonArray effects(ServerPlayer player) {
        JsonArray effects = new JsonArray();
        for (MobEffectInstance effect : player.getActiveEffects()) {
            JsonObject value = new JsonObject();
            value.addProperty("effect", effect.getEffect().getRegisteredName());
            value.addProperty("duration", effect.getDuration());
            value.addProperty("amplifier", effect.getAmplifier());
            value.addProperty("ambient", effect.isAmbient());
            effects.add(value);
        }
        return effects;
    }

    private JsonObject playerEvent(String type, Player player) {
        return event(type, pseudonymizer.actorId(player.getUUID()));
    }

    private void probeBlock(String actorId, Level level, BlockPos position) {
        pendingBlockProbes.add(new BlockProbe(
                actorId,
                level,
                position.immutable(),
                BuiltInRegistries.BLOCK.getKey(level.getBlockState(position).getBlock()).toString()));
    }

    private void appendBlockChanges(JsonArray events) {
        BlockProbe probe;
        while ((probe = pendingBlockProbes.poll()) != null) {
            String after = BuiltInRegistries.BLOCK
                    .getKey(probe.level().getBlockState(probe.position()).getBlock())
                    .toString();
            if (probe.blockBefore().equals(after)) {
                continue;
            }
            JsonObject event = event("block_changed", probe.actorId());
            addBlockContext(event, probe.level(), probe.position());
            event.addProperty("block_before", probe.blockBefore());
            event.addProperty("block_after", after);
            events.add(event);
        }
    }

    private static void addBlockContext(JsonObject event, Level level, BlockPos position) {
        event.addProperty("dimension", level.dimension().identifier().toString());
        event.addProperty("x", position.getX());
        event.addProperty("y", position.getY());
        event.addProperty("z", position.getZ());
    }

    private static JsonObject event(String type, String actorId) {
        JsonObject event = new JsonObject();
        event.addProperty("type", type);
        event.addProperty("actor_id", actorId);
        return event;
    }

    record Batch(long earliestSequence, long latestSequence, boolean gap, List<JsonObject> records) {
    }

    private record BlockProbe(
            String actorId,
            Level level,
            BlockPos position,
            String blockBefore) {
    }
}
