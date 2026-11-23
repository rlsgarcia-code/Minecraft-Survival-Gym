package dev.minecraftgym.server;

import java.io.IOException;
import java.nio.file.Path;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import net.fabricmc.api.ModInitializer;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerLifecycleEvents;
import net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents;
import net.fabricmc.fabric.api.event.player.AttackBlockCallback;
import net.fabricmc.fabric.api.event.player.PlayerBlockBreakEvents;
import net.fabricmc.fabric.api.event.player.UseBlockCallback;
import net.fabricmc.fabric.api.networking.v1.ServerPlayConnectionEvents;
import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.world.InteractionResult;

public final class MinecraftGymServerCollector implements ModInitializer {
    static final Logger LOGGER = LoggerFactory.getLogger("minecraft-gym-server-collector");

    @Override
    public void onInitialize() {
        try {
            Path config = FabricLoader.getInstance().getConfigDir();
            AuthoritativeStateCollector collector = new AuthoritativeStateCollector(
                    Pseudonymizer.loadOrCreate(config));
            int port = Integer.getInteger("minecraft.gym.collector.port", CollectorServer.DEFAULT_PORT);
            CollectorServer server = new CollectorServer(collector, port);

            ServerTickEvents.END_SERVER_TICK.register(collector::capture);
            ServerPlayConnectionEvents.JOIN.register(
                    (handler, sender, minecraftServer) -> collector.onJoin(handler.getPlayer()));
            ServerPlayConnectionEvents.DISCONNECT.register(
                    (handler, minecraftServer) -> collector.onDisconnect(handler.getPlayer()));
            PlayerBlockBreakEvents.AFTER.register(
                    (level, player, position, state, blockEntity) ->
                            collector.onBlockBreak(player, level, position, state));
            UseBlockCallback.EVENT.register((player, level, hand, hit) -> {
                collector.onBlockUse(player, level, hit.getBlockPos(), hit.getDirection());
                return InteractionResult.PASS;
            });
            AttackBlockCallback.EVENT.register((player, level, hand, position, direction) -> {
                collector.onBlockInteraction("block_attacked", player, level, position);
                return InteractionResult.PASS;
            });
            ServerLifecycleEvents.SERVER_STARTED.register(
                    minecraftServer -> {
                        server.start();
                        LOGGER.info("Passive collector listening on 127.0.0.1:{}", port);
                    });
            ServerLifecycleEvents.SERVER_STOPPING.register(minecraftServer -> server.close());
        } catch (IOException error) {
            throw new IllegalStateException("Could not initialize Minecraft Gym pseudonymization", error);
        }
    }
}
