namespace Cnn.LearningRateDecayers
{
    public class StepDecayer : LearningRateDecayer
    {
        private readonly int _step;
        private readonly double _rate;

        protected StepDecayer(double initial, int step, double rate) : base(initial)
        {
            _step = step;
            _rate = rate;
        }

        public override void Decay(int epoch)
        {
            if(epoch % _step == 0)
            {
                LearningRate *= _rate;
            }
        }
    }
}
