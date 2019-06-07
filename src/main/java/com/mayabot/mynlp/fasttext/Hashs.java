package com.mayabot.mynlp.fasttext;

public class Hashs {
    /**
     * String FNV-1a 32 bits Hash
     * from https://github.com/linkfluence/fastText4j/blob/master/src/main/java/fasttext/BaseDictionary.java
     * @param text
     * @return 返回无符号Int
     */
    public static long fnv1aHash(final String text) {
        // 0xffffffc5;
        int h = (int) 2166136261L;
        for (byte strByte : text.getBytes()) {
            h = (h ^ strByte) * 16777619;
        }
        return h & 0xffffffffL;
    }

    public static void main(String[] args) {
        System.out.println(fnv1aHash("中文"));
    }

}
