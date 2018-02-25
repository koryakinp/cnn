namespace Cnn.LearningRateDecayers
{
    public abstract class LearningRateDecayer
    {
        public double LearningRate { get; protected set; }

        protected LearningRateDecayer(double initial)
        {
            LearningRate = initial;
        }

        public abstract void Decay(int epoch);
    }
}
