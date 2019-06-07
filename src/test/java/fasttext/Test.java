package fasttext;

import com.mayabot.mynlp.fasttext.FastTextTrain;
import com.mayabot.mynlp.fasttext.TrainArgs;

public class Test {
    public static void main(String[] args) {
        FastTextTrain train = new FastTextTrain();

        new TrainArgs().setBucket(10000);
    }
}
