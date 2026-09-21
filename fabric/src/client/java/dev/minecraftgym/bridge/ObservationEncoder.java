package dev.minecraftgym.bridge;

import java.util.Base64;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.mojang.blaze3d.platform.NativeImage;

import net.minecraft.client.Minecraft;
import net.minecraft.client.Screenshot;
import net.minecraft.client.gui.screens.ChatScreen;
import net.minecraft.client.gui.screens.DeathScreen;
import net.minecraft.client.gui.screens.inventory.AbstractContainerScreen;
import net.minecraft.client.gui.screens.inventory.InventoryScreen;
import net.minecraft.client.player.LocalPlayer;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.phys.Vec3;

final class ObservationEncoder {
    private ObservationEncoder() {
    }

    static JsonObject capture(Minecraft minecraft, int width, int height, long worldSeed) {
        LocalPlayer player = minecraft.player;
        if (player == null || minecraft.level == null) {
            throw new IllegalStateException("A world and local player must be loaded");
        }

        JsonObject observation = new JsonObject();
        observation.add("rgb", captureRgb(minecraft, width, height));
        observation.add("vitals", captureVitals(player));
        observation.add("pose", capturePose(player));

        JsonArray inventoryIds = new JsonArray(36);
        JsonArray inventoryCounts = new JsonArray(36);
        for (int slot = 0; slot < 36; slot++) {
            ItemStack stack = player.getInventory().getItem(slot);
            inventoryIds.add(stack.isEmpty() ? 0 : BuiltInRegistries.ITEM.getId(stack.getItem()));
            inventoryCounts.add(stack.isEmpty() ? 0 : stack.getCount());
        }
        observation.add("inventory_ids", inventoryIds);
        observation.add("inventory_counts", inventoryCounts);
        observation.addProperty("equipped_slot", player.getInventory().selected);
        observation.addProperty("ui_mode", uiMode(minecraft));
        observation.addProperty("tick", minecraft.level.getGameTime());
        observation.addProperty("world_seed", worldSeed);
        return observation;
    }

    private static JsonObject captureRgb(Minecraft minecraft, int width, int height) {
        try (NativeImage source = Screenshot.takeScreenshot(minecraft.getMainRenderTarget());
                NativeImage resized = new NativeImage(width, height, false)) {
            source.resizeSubRectTo(0, 0, source.getWidth(), source.getHeight(), resized);
            byte[] rgb = new byte[width * height * 3];
            int offset = 0;
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    rgb[offset++] = resized.getRedOrLuminance(x, y);
                    rgb[offset++] = resized.getGreenOrLuminance(x, y);
                    rgb[offset++] = resized.getBlueOrLuminance(x, y);
                }
            }
            JsonObject image = new JsonObject();
            image.addProperty("encoding", "raw_rgb8");
            image.addProperty("width", width);
            image.addProperty("height", height);
            image.addProperty("data", Base64.getEncoder().encodeToString(rgb));
            return image;
        }
    }

    private static JsonArray captureVitals(LocalPlayer player) {
        JsonArray values = new JsonArray(8);
        values.add(clamp01(player.getHealth() / Math.max(1, player.getMaxHealth())));
        values.add(clamp01(player.getFoodData().getFoodLevel() / 20.0f));
        values.add(clamp01(player.getArmorValue() / 20.0f));
        values.add(clamp01(player.getAirSupply() / 300.0f));
        values.add(clamp01(player.experienceProgress));
        values.add(player.onGround() ? 1 : 0);
        values.add(player.isOnFire() ? 1 : 0);
        values.add(player.isUnderWater() ? 1 : 0);
        return values;
    }

    private static JsonArray capturePose(LocalPlayer player) {
        Vec3 velocity = player.getDeltaMovement();
        JsonArray values = new JsonArray(10);
        values.add(Math.tanh(player.getX() / 1024.0));
        values.add(clampSigned((player.getY() - 128.0) / 192.0));
        values.add(Math.tanh(player.getZ() / 1024.0));
        values.add(Math.tanh(velocity.x));
        values.add(Math.tanh(velocity.y));
        values.add(Math.tanh(velocity.z));
        values.add(clampSigned(player.getYRot() / 180.0));
        values.add(clampSigned(player.getXRot() / 90.0));
        values.add(player.onGround() ? 1 : -1);
        values.add(clampSigned(player.fallDistance / 40.0));
        return values;
    }

    private static int uiMode(Minecraft minecraft) {
        if (minecraft.screen == null) {
            return 0;
        }
        if (minecraft.screen instanceof InventoryScreen) {
            return 1;
        }
        if (minecraft.screen instanceof AbstractContainerScreen<?>) {
            return 2;
        }
        if (minecraft.screen instanceof ChatScreen) {
            return 3;
        }
        if (minecraft.screen instanceof DeathScreen) {
            return 4;
        }
        return 7;
    }

    private static float clamp01(float value) {
        return Math.max(0, Math.min(1, value));
    }

    private static double clampSigned(double value) {
        return Math.max(-1, Math.min(1, value));
    }
}
