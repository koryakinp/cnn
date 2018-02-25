namespace Cnn.LearningRateDecayers
{
    public class FlatDecayer : LearningRateDecayer
    {
        public FlatDecayer(double initial) : base(initial) {}

        public override void Decay(int epoch) {}
    }
}
