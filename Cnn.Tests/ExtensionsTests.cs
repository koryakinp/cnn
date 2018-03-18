using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Text;

namespace Cnn.Tests
{
    [TestClass]
    public class ExtensionsTests
    {
        [TestMethod]
        public void GetSliceTest()
        {
            var init = new double[3,3,3]
            {
                {
                    { 1,1,1 },
                    { 1,1,1 },
                    { 1,1,1 }
                },
                {
                    { 2,2,2 },
                    { 2,2,2 },
                    { 2,2,2 }
                },
                {
                    { 3,3,3 },
                    { 3,3,3 },
                    { 3,3,3 }
                }
            };

            var expected1 = new double[3, 3]
            {
                { 1,1,1 },
                { 1,1,1 },
                { 1,1,1 }
            };

            var expected2 = new double[3, 3]
            {
                { 2,2,2 },
                { 2,2,2 },
                { 2,2,2 }
            };

            var expected3 = new double[3, 3]
            {
                { 3,3,3 },
                { 3,3,3 },
                { 3,3,3 }
            };

            init.GetSlice(0)
                .ForEach((q, i, j) => Assert.AreEqual(expected1[i, j], q));

            init.GetSlice(1)
                .ForEach((q, i, j) => Assert.AreEqual(expected2[i, j], q));

            init.GetSlice(2)
                .ForEach((q, i, j) => Assert.AreEqual(expected3[i, j], q));
        }
    }
}
