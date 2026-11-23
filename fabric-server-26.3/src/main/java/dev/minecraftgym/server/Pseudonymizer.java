package dev.minecraftgym.server;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.PosixFilePermission;
import java.security.GeneralSecurityException;
import java.security.SecureRandom;
import java.util.HexFormat;
import java.util.Set;
import java.util.UUID;

import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;

final class Pseudonymizer {
    private static final String ALGORITHM = "HmacSHA256";
    private final byte[] key;

    private Pseudonymizer(byte[] key) {
        this.key = key.clone();
    }

    static Pseudonymizer loadOrCreate(Path configDirectory) throws IOException {
        Files.createDirectories(configDirectory);
        Path keyPath = configDirectory.resolve("minecraft-gym-collector.key");
        byte[] key;
        if (Files.isRegularFile(keyPath)) {
            key = HexFormat.of().parseHex(Files.readString(keyPath).trim());
            if (key.length != 32) {
                throw new IOException("Collector pseudonym key must contain 32 bytes");
            }
        } else {
            key = new byte[32];
            new SecureRandom().nextBytes(key);
            Files.writeString(keyPath, HexFormat.of().formatHex(key));
            try {
                Files.setPosixFilePermissions(
                        keyPath,
                        Set.of(PosixFilePermission.OWNER_READ, PosixFilePermission.OWNER_WRITE));
            } catch (UnsupportedOperationException ignored) {
                // Windows and non-POSIX filesystems do not expose these permissions.
            }
        }
        return new Pseudonymizer(key);
    }

    String actorId(UUID playerId) {
        try {
            Mac mac = Mac.getInstance(ALGORITHM);
            mac.init(new SecretKeySpec(key, ALGORITHM));
            byte[] digest = mac.doFinal(playerId.toString().getBytes(java.nio.charset.StandardCharsets.UTF_8));
            return "actor:pseudonym:" + HexFormat.of().formatHex(digest, 0, 16);
        } catch (GeneralSecurityException error) {
            throw new IllegalStateException("HMAC-SHA256 is unavailable", error);
        }
    }
}
