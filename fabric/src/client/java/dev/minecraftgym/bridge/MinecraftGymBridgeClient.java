package dev.minecraftgym.bridge;

import java.util.concurrent.atomic.AtomicBoolean;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientLifecycleEvents;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;

public final class MinecraftGymBridgeClient implements ClientModInitializer {
    @Override
    public void onInitializeClient() {
        BridgeRuntime runtime = new BridgeRuntime();
        BridgeServer server = new BridgeServer(runtime, 25570);
        String quickPlayWorld = System.getenv("MINECRAFT_GYM_QUICK_PLAY_WORLD");
        AtomicBoolean quickPlayRequested = new AtomicBoolean();
        ClientTickEvents.START_CLIENT_TICK.register(client -> {
            if (quickPlayWorld != null
                    && !quickPlayWorld.isBlank()
                    && client.level == null
                    && client.isGameLoadFinished()
                    && quickPlayRequested.compareAndSet(false, true)) {
                BridgeLog.LOGGER.info("Opening development smoke-test world {}", quickPlayWorld);
                client.createWorldOpenFlows().openWorld(quickPlayWorld, () -> {
                    BridgeLog.LOGGER.error("Could not open development smoke-test world {}", quickPlayWorld);
                });
            }
        });
        ClientTickEvents.START_CLIENT_TICK.register(runtime::onStartTick);
        ClientTickEvents.END_CLIENT_TICK.register(runtime::onEndTick);
        ClientLifecycleEvents.CLIENT_STOPPING.register(client -> server.close());
        server.start();
        BridgeLog.LOGGER.info("Minecraft Gym Bridge listening on 127.0.0.1:25570");
    }
}
