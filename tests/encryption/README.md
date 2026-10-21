<!--
  ~ Licensed to the Apache Software Foundation (ASF) under one
  ~ or more contributor license agreements.  See the NOTICE file
  ~ distributed with this work for additional information
  ~ regarding copyright ownership.  The ASF licenses this file
  ~ to you under the Apache License, Version 2.0 (the
  ~ "License"); you may not use this file except in compliance
  ~ with the License.  You may obtain a copy of the License at
  ~
  ~   http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing,
  ~ software distributed under the License is distributed on an
  ~ "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  ~ KIND, either express or implied.  See the License for the
  ~ specific language governing permissions and limitations
  ~ under the License.
-->

# AGS1 test vectors

`JAVA_STREAMS` in `test_stream.py` holds AGS1 streams written by Java's
`AesGcmOutputFile` (Apache Iceberg 1.11.0), not by PyIceberg. They exist so
PyIceberg's AGS1 support is checked against another implementation's bytes rather
than only against its own round trip.

| Vector | Encrypted size | Plaintext | Pins |
| --- | --- | --- | --- |
| `empty` | 36 B | 0 B | Java writes an 8 byte header **plus one empty block** for an empty file, not a bare header |
| `partial-block` | 136 B | 100 B | Header, nonce/tag layout, and a single short block |
| `partial-block-no-aad` | 136 B | 100 B | The same stream with a null AAD prefix, so the block index alone is the AAD |

Multi-block vectors are left out: the 1 MiB block size is hard-coded, so each
extra block adds 1 MiB of incompressible ciphertext. Moving cross-client vectors
to iceberg-verification is tracked in apache/iceberg-python#4010.

## Parameters

Every vector uses:

- **Key**: 16 bytes, `0x00` through `0x0f`
- **AAD prefix**: ASCII `pyiceberg-ags1`, except `partial-block-no-aad`, which has none
- **Plaintext**: byte `i` is `i % 251`

## Regenerating

Each block uses a fresh random nonce, so regenerating produces different bytes.
The lengths, and the plaintext each vector decrypts to, are deterministic. The
tests decrypt the vectors rather than compare them byte for byte, so regenerated
vectors can replace the ones in `test_stream.py` as long as the parameters above
are unchanged.

Save the program below as `GenerateAgs1Vectors.java`, then run it from the same
directory with JDK 17 or later. It prints one line of hex per vector, ready to
paste into `JAVA_STREAMS`.

<!-- markdown-link-check-disable -->
```bash
V=1.11.0
for a in iceberg-core iceberg-api iceberg-bundled-guava; do
  curl -sfLO "https://repo1.maven.org/maven2/org/apache/iceberg/$a/$V/$a-$V.jar"
done
java -cp "iceberg-api-$V.jar:iceberg-bundled-guava-$V.jar:iceberg-core-$V.jar" \
  GenerateAgs1Vectors.java
rm iceberg-*-$V.jar
```
<!-- markdown-link-check-enable -->

```java
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.HexFormat;
import org.apache.iceberg.encryption.AesGcmOutputFile;
import org.apache.iceberg.io.PositionOutputStream;

public class GenerateAgs1Vectors {
  static final byte[] KEY = new byte[16];
  static final byte[] AAD_PREFIX = "pyiceberg-ags1".getBytes(StandardCharsets.UTF_8);

  static {
    for (int i = 0; i < KEY.length; i++) {
      KEY[i] = (byte) i;
    }
  }

  static byte[] plaintext(int length) {
    byte[] out = new byte[length];
    for (int i = 0; i < length; i++) {
      out[i] = (byte) (i % 251);
    }
    return out;
  }

  static void print(File dir, String name, int length, byte[] aadPrefix) throws IOException {
    File target = new File(dir, name + ".ags1");
    AesGcmOutputFile encrypted =
        new AesGcmOutputFile(org.apache.iceberg.Files.localOutput(target), KEY, aadPrefix);
    try (PositionOutputStream stream = encrypted.create()) {
      stream.write(plaintext(length));
    }
    System.out.println(name + ": " + HexFormat.of().formatHex(Files.readAllBytes(target.toPath())));
  }

  public static void main(String[] args) throws IOException {
    File dir = Files.createTempDirectory("ags1").toFile();
    print(dir, "empty", 0, AAD_PREFIX);
    print(dir, "partial-block", 100, AAD_PREFIX);
    print(dir, "partial-block-no-aad", 100, null);
  }
}
```
